from __future__ import annotations

import datetime
from functools import partial
from pathlib import Path
from typing import cast
from typing import Literal

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jaxtyping import Array
from jaxtyping import Float
from omegaconf import DictConfig
from play_lmp import EpisodeBatch
from play_lmp import make_train_step
from play_lmp import PlayLMP
from play_lmp import preprocess_image
from play_lmp import preprocess_proprio
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    dataset = get_dataset(cfg, "train")
    rgb_stats, proprio_stats = get_normalization_stats(cfg)
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, model_key)
    print(f"Total trainable parameters: {num_model_parameters(model):_}")
    optimizer: optax.GradientTransformation = hydra.utils.instantiate(
        cfg.training.optimizer
    )
    train(
        cfg.training,
        model,
        optimizer,
        dataset,
        rgb_stats,
        proprio_stats,
        key=random_key,
    )


def get_model(cfg: DictConfig, key: jax.Array) -> PlayLMP:
    cnn_key, plan_rec_key, plan_proposal_key, policy_key = jax.random.split(key, 4)
    cnn = hydra.utils.instantiate(cfg.cnn)(key=cnn_key)
    plan_recognizer = hydra.utils.instantiate(cfg.plan_recognizer)(
        cnn=cnn, key=plan_rec_key
    )
    plan_proposal = hydra.utils.instantiate(cfg.plan_proposal)(
        cnn=cnn, key=plan_proposal_key
    )
    policy = hydra.utils.instantiate(cfg.policy)(cnn=cnn, key=policy_key)
    model = PlayLMP(plan_recognizer, plan_proposal, policy)
    return model


def train(
    cfg: DictConfig,
    model: PlayLMP,
    optimizer: optax.GradientTransformation,
    dataset: tf.data.Dataset,
    rgb_normalization_stats: Array,
    proprio_normalization_stats: Array,
    key: jax.Array,
) -> None:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    tb_writer = tf.summary.create_file_writer(
        datetime.datetime.now().strftime(cfg.tensorboard_log_dir)
    )
    with tb_writer.as_default():
        for step, batch in zip(range(cfg.num_steps), dataset):
            key, step_key = jax.random.split(key)
            episode_batch = tfds_batch_to_episode_batch(
                batch, (128, 128), rgb_normalization_stats, proprio_normalization_stats
            )
            model, opt_state, loss = eqx.filter_jit(make_train_step)(
                model,
                optimizer,
                opt_state,
                episode_batch,
                step_key,
                method=cfg.method,
                beta=cfg.beta,
            )
            tf.summary.scalar("loss", float(loss), step=step)
            tb_writer.flush()
            print(f"Step {step}: Training loss {loss}")


def num_model_parameters(model: eqx.Module) -> int:
    filtered_model = eqx.filter(model, eqx.is_array)
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(filtered_model))


def tfds_batch_to_episode_batch(
    batch: dict,
    target_image_size: tuple[int, int],
    rgb_stats: Float[Array, "2 channel"],
    proprio_stats: Float[Array, "2 d_proprio"],
) -> EpisodeBatch:
    rgb_observations = jnp.asarray(batch["observation"]["rgb"])
    rgb_observations = jax.jit(
        jax.vmap(
            lambda sequence: jax.lax.map(
                partial(
                    preprocess_image,
                    target_size=target_image_size + (3,),
                    channel_mean=rgb_stats[0],
                    channel_std=rgb_stats[1],
                ),
                sequence,
                batch_size=8,
            )
        )
    )(rgb_observations)
    proprio_observations = jnp.asarray(batch["observation"]["effector_translation"])
    proprio_observations = jax.jit(
        jax.vmap(
            jax.vmap(
                partial(preprocess_proprio, mean=proprio_stats[0], std=proprio_stats[1])
            )
        )
    )(proprio_observations)
    actions = jnp.asarray(batch["action"])
    # The "valid" key is introduced by `pad_to_cardinality`
    episode_lengths = jnp.asarray(batch["valid"]).sum(axis=1)
    return EpisodeBatch(
        rgb_observations=rgb_observations,
        proprio_observations=proprio_observations,
        actions=actions,
        episode_lengths=episode_lengths,
    )


def get_dataset(
    cfg: DictConfig, split: Literal["train", "val", "test"], batch: bool = True
) -> tf.data.Dataset:
    builder = tfds.builder_from_directory(cfg.training.tf_dataset_path)
    all_dataset = cast(tf.data.Dataset, builder.as_dataset(split="train"))
    num_val_episodes = round(len(all_dataset) * cfg.training.val_split_proportion)
    num_test_episodes = round(len(all_dataset) * cfg.training.test_split_proportion)
    print(
        f"Training episodes: {len(all_dataset) - num_val_episodes - num_test_episodes}"
    )
    print(f"Validation episodes: {num_val_episodes}")
    print(f"Testing episodes: {num_test_episodes}")
    splits = {
        "val": all_dataset.take(num_val_episodes),
        "test": all_dataset.skip(num_val_episodes).take(num_test_episodes),
        "train": all_dataset.skip(num_val_episodes + num_test_episodes),
    }
    dataset = splits[split]
    if not batch:
        return dataset
    return (
        dataset.repeat()
        .shuffle(buffer_size=cfg.training.shuffle_buffer_size, seed=cfg.random_seed)
        .flat_map(
            lambda x: x["steps"]
            .take(cfg.training.sequence_padding_length)
            .apply(
                tf.data.experimental.pad_to_cardinality(
                    cfg.training.sequence_padding_length
                )
            )
        )
        .batch(cfg.training.sequence_padding_length)
        .batch(cfg.training.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_normalization_stats(
    cfg: DictConfig,
) -> tuple[Float[Array, "2 3"], Float[Array, "2 d_proprio"]]:
    stats_path = Path(cfg.training.normalization_stats_path)
    if stats_path.exists():
        stats = jnp.load(stats_path)
        return stats[:, :3], stats[:, 3:]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = (
        get_dataset(cfg, "train", batch=False)
        .flat_map(lambda x: x["steps"])
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )
    rgb_batch_means, proprio_batch_means = [], []
    for batch in tqdm(dataset.as_numpy_iterator(), desc="Calculating observation mean"):
        rgb = jnp.asarray(batch["observation"]["rgb"]) / 255  # type: ignore
        proprio = jnp.asarray(batch["observation"]["effector_translation"])  # type: ignore
        rgb_batch_means.append(rgb.mean(axis=(0, 1, 2)))
        proprio_batch_means.append(proprio.mean(axis=0))
    rgb_mean, proprio_mean = [
        jnp.mean(jnp.stack(arr), axis=0)
        for arr in [rgb_batch_means, proprio_batch_means]
    ]
    rgb_batch_variances, proprio_batch_variances = [], []
    for batch in tqdm(
        dataset.as_numpy_iterator(), desc="Calculating observation stddev"
    ):
        rgb = jnp.asarray(batch["observation"]["rgb"]) / 255  # type: ignore
        proprio = jnp.asarray(batch["observation"]["effector_translation"])  # type: ignore
        rgb_batch_variances.append(jnp.square(rgb - rgb_mean).mean(axis=(0, 1, 2)))
        proprio_batch_variances.append(jnp.square(proprio - proprio_mean).mean(axis=0))
    rgb_std, proprio_std = [
        jnp.sqrt(jnp.mean(jnp.stack(arr), axis=0))
        for arr in [rgb_batch_variances, proprio_batch_variances]
    ]
    rgb_stats = jnp.stack([rgb_mean, rgb_std])
    proprio_stats = jnp.stack([proprio_mean, proprio_std])
    jnp.save(stats_path, jnp.hstack([rgb_stats, proprio_stats]))
    return rgb_stats, proprio_stats


if __name__ == "__main__":
    main()
