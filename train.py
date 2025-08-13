import datetime
from pathlib import Path

import equinox as eqx
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import jmp
import minari
import optax
import tensorflow as tf
from jaxtyping import Array
from jaxtyping import Float
from omegaconf import DictConfig
from play_lmp import EpisodeBatch
from play_lmp import make_train_step
from play_lmp import PlayLMP
from play_lmp import preprocess_action
from play_lmp import preprocess_goal
from play_lmp import preprocess_observation
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    dataset, env = get_dataset_and_env(cfg)
    observation_stats, goal_stats, action_stats = get_normalization_stats(cfg)
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
    preprocessed_dataset = preprocess_dataset(
        dataset, observation_stats, goal_stats, action_stats, target_action_range
    )
    train(
        cfg.training,
        model,
        optimizer,
        preprocessed_dataset,
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
    dataset: EpisodeBatch,
    mp_policy: jmp.Policy,
    key: jax.Array,
) -> None:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = mp_policy.cast_to_param(opt_state)
    tb_writer = tf.summary.create_file_writer(
        datetime.datetime.now().strftime(cfg.tensorboard_log_dir)
    )
    with tb_writer.as_default():
        for step in range(cfg.num_steps):
            key, step_key = jax.random.split(key)
            batch = get_batch(
                cfg,
                dataset,
                step_key,
            )
            key, step_key = jax.random.split(key)
            model, opt_state, loss = eqx.filter_jit(make_train_step)(
                model,
                optimizer,
                opt_state,
                mp_policy,
                batch,
                step_key,
                method=cfg.method,
                beta=cfg.beta,
            )
            tf.summary.scalar("step_loss/train", float(loss), step=step)
            tb_writer.flush()
            print(f"Step {step}: Training loss {loss}")


def num_model_parameters(model: eqx.Module) -> int:
    filtered_model = eqx.filter(model, eqx.is_array)
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(filtered_model))


def get_batch(
    cfg: DictConfig,
    dataset: EpisodeBatch,
    key: jax.Array,
) -> EpisodeBatch:
    key, sampling_key = jax.random.split(key)
    episode_indices = jax.random.randint(
        sampling_key, (cfg.batch_size,), 0, dataset.achieved_goals.shape[0]
    )
    observations = dataset.observations[episode_indices]
    achieved_goals = dataset.achieved_goals[episode_indices]
    actions = dataset.actions[episode_indices]
    episode_lengths = dataset.episode_lengths[episode_indices]
    key, sampling_key = jax.random.split(key)
    min_episode_length, max_episode_length = (
        (cfg.gcbc_window_length_min, cfg.gcbc_window_length_max)
        if cfg.method == "play-gcbc"
        else 2 * (cfg.lmp_window_length,)
    )
    start_indices = jax.random.randint(
        sampling_key, (cfg.batch_size,), 0, episode_lengths - max_episode_length
    )
    key, sampling_key = jax.random.split(key)
    episode_lengths = jnp.minimum(
        episode_lengths - start_indices,
        jax.random.randint(
            sampling_key, (cfg.batch_size,), min_episode_length, max_episode_length + 1
        ),
    )

    @jax.jit
    def take_slice(arr: Array) -> Array:
        indices = start_indices.reshape(-1, 1) + jnp.arange(max_episode_length)
        return arr[jnp.arange(arr.shape[0]).reshape(-1, 1), indices]

    return EpisodeBatch(
        take_slice(observations),
        take_slice(achieved_goals),
        take_slice(actions),
        episode_lengths,
    )


def pack_goals(goals: dict) -> Array:
    keys = ["bottom burner", "kettle", "light switch", "microwave"]
    return jnp.concat([goals[key] for key in keys], axis=1)


def preprocess_dataset(
    dataset: minari.MinariDataset,
    observation_stats: Array,
    goal_stats: Array,
    action_stats: Array,
    target_action_range: Array,
) -> EpisodeBatch:
    padding_length = 1024
    observations, achieved_goals, actions, episode_lengths = [], [], [], []
    for episode in tqdm(dataset, desc="Preprocessing dataset episodes"):
        length = episode.truncations.size
        episode_lengths.append(length)
        observations.append(
            jnp.pad(
                episode.observations["observation"][:-1],
                ((0, padding_length - length), (0, 0)),
            )
        )
        achieved_goals.append(
            jnp.pad(
                pack_goals(episode.observations["achieved_goal"])[:-1],
                ((0, padding_length - length), (0, 0)),
            )
        )
        actions.append(
            jnp.pad(
                episode.actions,
                ((0, padding_length - length), (0, 0)),
            )
        )
    observations = jnp.stack(observations)
    achieved_goals = jnp.stack(achieved_goals)
    actions = jnp.stack(actions)
    observations = jax.jit(
        jax.vmap(
            jax.vmap(preprocess_observation, in_axes=(0, None, None)),
            in_axes=(0, None, None),
        )
    )(observations, observation_stats[0], observation_stats[1])
    actions = jax.jit(
        jax.vmap(
            jax.vmap(preprocess_action, in_axes=(0, None, None, None, None)),
            in_axes=(0, None, None, None, None),
        )
    )(
        actions,
        action_stats[0],
        action_stats[1],
        target_action_range[0],
        target_action_range[1],
    )
    achieved_goals = jax.vmap(
        jax.vmap(preprocess_goal, in_axes=(0, None, None)),
        in_axes=(0, None, None),
    )(achieved_goals, goal_stats[0], goal_stats[1])
    return EpisodeBatch(
        observations, achieved_goals, actions, jnp.asarray(episode_lengths)
    )


def get_dataset_and_env(
    cfg: DictConfig,
) -> tuple[minari.MinariDataset, gym.Env]:
    dataset = minari.load_dataset("D4RL/kitchen/mixed-v2", download=True)
    dataset.set_seed(cfg.random_seed)
    env = dataset.recover_environment(render_mode="rgb_array")
    return dataset, env


def get_normalization_stats(
    cfg: DictConfig,
) -> tuple[
    Float[Array, "2 d_obs"], Float[Array, "2 d_goal"], Float[Array, "2 d_action"]
]:
    stats_path = Path(cfg.training.normalization_stats_path)
    if stats_path.exists():
        stats = jnp.load(stats_path)
        return (
            jnp.asarray(stats["observation"]),
            jnp.asarray(stats["goal"]),
            jnp.asarray(stats["action"]),
        )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    dataset, _ = get_dataset_and_env(cfg)
    observation_episode_means, goal_episode_means = [], []
    action_episode_max, action_episode_min = [], []
    for episode in tqdm(dataset, desc="Calculating mean"):
        observations = jnp.asarray(episode.observations["observation"])
        actions = jnp.asarray(episode.actions)
        goals = jnp.concat(episode.observations["achieved_goal"].values(), axis=1)
        observation_episode_means.append(observations.mean(axis=0))
        goal_episode_means.append(goals.mean(axis=0))
        action_episode_max.append(actions.max(axis=0))
        action_episode_min.append(actions.min(axis=0))
    observation_mean = jnp.mean(jnp.stack(observation_episode_means), axis=0)
    goal_mean = jnp.mean(jnp.stack(goal_episode_means), axis=0)
    action_max = jnp.max(jnp.stack(action_episode_max), axis=0)
    action_min = jnp.min(jnp.stack(action_episode_min), axis=0)
    action_stats = jnp.stack([action_max, action_min])
    del (
        observation_episode_means,
        goal_episode_means,
        action_episode_max,
        action_episode_min,
    )
    observation_episode_variances, goal_episode_variances = [], []
    for episode in tqdm(dataset, desc="Calculating stddev"):
        observations = jnp.asarray(episode.observations["observation"])
        goals = jnp.concat(episode.observations["achieved_goal"].values(), axis=1)
        observation_episode_variances.append(
            jnp.square(observations - observation_mean).mean(axis=0)
        )
        goal_episode_variances.append(jnp.square(goals - goal_mean).mean(axis=0))
    observation_std = jnp.sqrt(
        jnp.mean(jnp.stack(observation_episode_variances), axis=0)
    )
    goal_std = jnp.sqrt(jnp.mean(jnp.stack(goal_episode_variances), axis=0))
    observation_stats = jnp.stack([observation_mean, observation_std])
    goal_stats = jnp.stack([goal_mean, goal_std])
    jnp.savez(
        stats_path, observation=observation_stats, action=action_stats, goal=goal_stats
    )
    return observation_stats, goal_stats, action_stats


if __name__ == "__main__":
    main()
