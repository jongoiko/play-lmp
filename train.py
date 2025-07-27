from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast
from typing import Literal

import hydra
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Int8
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    dataset = get_dataset(cfg, "train")
    rgb_stats, proprio_stats = get_normalization_stats(cfg)
    for batch in dataset:
        episode_batch = EpisodeBatch.from_tfds_batch(batch)  # type: ignore
        # TODO: Training loop on episode_batch
        del episode_batch
    del random_key


@dataclasses.dataclass
class EpisodeBatch:
    rgb_observations: Int8[Array, "batch time height width channel"]
    proprio_observations: Float[Array, "batch time d_proprio"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]

    @staticmethod
    def from_tfds_batch(batch: dict) -> EpisodeBatch:
        rgb_observations = jnp.asarray(batch["observation"]["rgb"])
        proprio_observations = jnp.asarray(batch["observation"]["effector_translation"])
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
            lambda x: x["steps"].apply(
                tf.data.experimental.pad_to_cardinality(
                    cfg.training.sequence_padding_length
                )
            ),
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
