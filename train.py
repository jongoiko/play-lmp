import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jmp
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm

with install_import_hook("play_lmp", "beartype.beartype"):
    from play_lmp.play_lmp import EpisodeBatch
    from play_lmp.play_lmp import make_train_step
    from play_lmp.play_lmp import eval_loss
    from play_lmp.play_lmp import PlayLMP
    from play_lmp.preprocessing import preprocess_action
    from play_lmp.preprocessing import preprocess_observation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    train_dataset = get_dataset(cfg, fold="training")
    val_dataset = get_dataset(cfg, fold="validation")
    observation_stats, action_stats = get_normalization_stats(cfg, train_dataset)
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, cfg.training.method, model_key)
    print(f"Total trainable parameters: {num_model_parameters(model):_}")
    mp_policy = jmp.get_policy(cfg.training.mixed_precision_policy)
    model = mp_policy.cast_to_param(model)
    optimizer: optax.GradientTransformation = hydra.utils.instantiate(
        cfg.training.optimizer
    )
    target_action_range = jnp.stack(
        [
            hydra.utils.instantiate(cfg.model.target_action_max),
            hydra.utils.instantiate(cfg.model.target_action_min),
        ]
    )
    preprocessed_train_dataset = preprocess_dataset(
        train_dataset, observation_stats, action_stats, target_action_range
    )
    preprocessed_val_dataset = preprocess_dataset(
        val_dataset, observation_stats, action_stats, target_action_range
    )
    train(
        cfg,
        model,
        optimizer,
        preprocessed_train_dataset,
        preprocessed_val_dataset,
        mp_policy,
        key=random_key,
    )


def get_model(cfg: DictConfig, method: str, key: jax.Array) -> PlayLMP:
    if method != "play-lmp":
        cfg.d_latent = 0
    plan_rec_key, plan_proposal_key, policy_key = jax.random.split(key, 3)
    plan_recognizer = hydra.utils.instantiate(cfg.plan_recognizer)(key=plan_rec_key)
    plan_proposal = hydra.utils.instantiate(cfg.plan_proposal)(key=plan_proposal_key)
    policy = hydra.utils.instantiate(cfg.policy)(key=policy_key)
    model = PlayLMP(plan_recognizer, plan_proposal, policy)
    return model


def train(
    cfg: DictConfig,
    model: PlayLMP,
    optimizer: optax.GradientTransformation,
    train_dataset: EpisodeBatch,
    val_dataset: EpisodeBatch,
    mp_policy: jmp.Policy,
    key: jax.Array,
) -> None:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = mp_policy.cast_to_param(opt_state)
    tb_writer = SummaryWriter(
        datetime.datetime.now().strftime(cfg.training.tensorboard_log_dir)
    )
    for step in range(cfg.training.num_steps):
        key, step_key = jax.random.split(key)
        batch = get_batch(cfg.training, train_dataset, step_key)
        key, step_key = jax.random.split(key)
        model, opt_state, loss, stats = eqx.filter_jit(make_train_step)(
            model,
            optimizer,
            opt_state,
            mp_policy,
            batch,
            step_key,
            method=cfg.training.method,
            beta=cfg.training.beta,
        )
        tb_writer.add_scalar("step_loss/train", float(loss), step)
        for stats_key, value in stats.items():
            tb_writer.add_scalar(f"step_{stats_key}/train", float(value), step)
        print(f"Step {step}: Training loss {loss}")
        tb_writer.flush()
        if step % cfg.training.evaluation.every_n_steps == 0:
            eqx.tree_serialise_leaves(cfg.training.evaluation.model_save_path, model)
            inference_model = eqx.nn.inference_mode(model)
            losses, stats = [], {}
            for _ in tqdm(
                range(cfg.training.evaluation.num_val_batches),
                desc="Estimating validation loss",
            ):
                key, step_key = jax.random.split(key)
                batch = get_batch(cfg.training, val_dataset, step_key)
                loss, batch_stats = eqx.filter_jit(eval_loss)(
                    inference_model,
                    mp_policy,
                    batch,
                    step_key,
                    cfg.training.method,
                    cfg.training.beta,
                )
                losses.append(loss)
                for stats_key, value in batch_stats.items():
                    stats[stats_key] = stats.get(stats_key, []) + [value]
            val_loss = jnp.asarray(losses).mean()
            for stats_key, value in stats.items():
                tb_writer.add_scalar(
                    f"step_{stats_key}/val", jnp.asarray(value).mean(), step
                )
            tb_writer.add_scalar("step_loss/val", float(val_loss), step)
            print(f"Step {step}: Validation loss {val_loss}")


def num_model_parameters(model: eqx.Module) -> int:
    filtered_model = eqx.filter(model, eqx.is_array)
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(filtered_model))


@partial(jax.jit, static_argnums=[0])
def get_batch(
    cfg: DictConfig,
    dataset: EpisodeBatch,
    key: jax.Array,
) -> EpisodeBatch:
    key, sampling_key = jax.random.split(key)
    all_episode_lengths = dataset.episode_lengths
    episode_indices = jax.random.choice(
        sampling_key,
        int(all_episode_lengths.shape[0]),
        (cfg.batch_size,),
        p=all_episode_lengths / all_episode_lengths.sum(),
    )
    observations = dataset.observations[episode_indices]
    actions = dataset.actions[episode_indices]
    episode_lengths = dataset.episode_lengths[episode_indices]
    min_episode_length, max_episode_length = (
        (cfg.gcbc_window_length_min, cfg.gcbc_window_length_max)
        if cfg.method == "play-gcbc"
        else 2 * (cfg.lmp_window_length,)
    )
    key, sampling_key = jax.random.split(key)
    window_lengths = jax.random.randint(
        sampling_key, (cfg.batch_size,), min_episode_length, max_episode_length + 1
    )
    key, sampling_key = jax.random.split(key)
    start_indices = jax.random.randint(
        sampling_key, (cfg.batch_size,), 0, episode_lengths - window_lengths
    )
    episode_lengths = jnp.minimum(episode_lengths - start_indices, window_lengths)

    def take_slice(arr: Array) -> Array:
        indices = start_indices.reshape(-1, 1) + jnp.arange(max_episode_length)
        return arr[jnp.arange(arr.shape[0]).reshape(-1, 1), indices]

    return EpisodeBatch(
        take_slice(observations),
        take_slice(actions),
        episode_lengths,
    )


def preprocess_dataset(
    dataset: EpisodeBatch,
    observation_stats: Array,
    action_stats: Array,
    target_action_range: Array,
) -> EpisodeBatch:
    observations = jax.jit(
        jax.vmap(
            jax.vmap(preprocess_observation, in_axes=(0, None, None)),
            in_axes=(0, None, None),
        )
    )(dataset.observations, observation_stats[0], observation_stats[1])
    actions = jax.jit(
        jax.vmap(
            jax.vmap(preprocess_action, in_axes=(0, None, None, None, None)),
            in_axes=(0, None, None, None, None),
        )
    )(
        dataset.actions,
        action_stats[0],
        action_stats[1],
        target_action_range[0],
        target_action_range[1],
    )
    return EpisodeBatch(observations, actions, dataset.episode_lengths)


def get_dataset(
    cfg: DictConfig, fold: Literal["training", "validation"]
) -> EpisodeBatch:
    dataset_path = Path(cfg.training.dataset_path) / fold
    start_end_ids = jnp.load(dataset_path / "ep_start_end_ids.npy")
    errors = 0
    actions_key = "rel_actions" if cfg.training.relative_actions else "actions"
    episode_actions, episode_observations = [], []
    for episode_num, (start_id, end_id) in tqdm(
        enumerate(start_end_ids), desc=f"Reading {fold} data"
    ):
        actions, observations = [], []
        for id in tqdm(
            range(start_id, end_id + 1),
            desc=f"Reading episode {episode_num}",
            leave=False,
        ):
            try:
                step = jnp.load(dataset_path / f"episode_{str(id).zfill(7)}.npz")
                actions.append(step[actions_key])
                observations.append(
                    jnp.concatenate([step["robot_obs"], step["scene_obs"]])
                )
            except Exception as _:
                errors += 1
        episode_actions.append(jnp.stack(actions))
        episode_observations.append(jnp.stack(observations))
    episode_lengths = [obs.shape[0] for obs in episode_observations]
    max_episode_length = max(episode_lengths)
    print(f"A total of {errors} steps could not be read.")
    observations = jnp.stack(
        [
            jnp.pad(obs, ((0, max_episode_length - obs.shape[0]), (0, 0)))
            for obs in episode_observations
        ]
    )
    actions = jnp.stack(
        [
            jnp.pad(action, ((0, max_episode_length - action.shape[0]), (0, 0)))
            for action in episode_actions
        ]
    )
    return EpisodeBatch(observations, actions, jnp.asarray(episode_lengths))


def get_normalization_stats(
    cfg: DictConfig,
    dataset: EpisodeBatch | None,
) -> tuple[Float[Array, "2 d_obs"], Float[Array, "2 d_action"]]:
    stats_path = Path(cfg.training.normalization_stats_path)
    if stats_path.exists():
        stats = jnp.load(stats_path)
        return (
            jnp.asarray(stats["observation"]),
            jnp.asarray(stats["action"]),
        )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    assert dataset is not None
    observations, actions = [], []
    for episode_idx, episode_length in enumerate(dataset.episode_lengths):
        observations.append(dataset.observations[episode_idx, :episode_length])
        actions.append(dataset.actions[episode_idx, :episode_length])
    observations = jnp.concatenate(observations)
    actions = jnp.concatenate(actions)
    observation_stats = jnp.stack(
        [
            observations.mean(axis=0),
            observations.std(axis=0),
        ]
    )
    action_stats = jnp.stack(
        [
            actions.max(axis=0),
            actions.min(axis=0),
        ]
    )
    jnp.savez(stats_path, observation=observation_stats, action=action_stats)
    return observation_stats, action_stats


if __name__ == "__main__":
    main()
