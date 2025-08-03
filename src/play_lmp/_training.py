import dataclasses
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

from ._play_lmp import PlayLMP


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EpisodeBatch:
    rgb_observations: Float[Array, "batch time height width channel"]
    proprio_observations: Float[Array, "batch time d_proprio"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]


def make_train_step(
    model: PlayLMP,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    batch: EpisodeBatch,
    key: jax.Array,
    method: Literal["play-lmp", "play-gcbc"],
    beta: float = 0.0,
) -> tuple[PlayLMP, PyTree, Float[Array, ""]]:
    if method == "play-gcbc":
        loss_value, grads = eqx.filter_value_and_grad(play_gcbc_loss)(model, batch)
    elif method == "play-lmp":
        loss_value, grads = eqx.filter_value_and_grad(play_lmp_loss)(
            model, batch, key, beta
        )
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def play_lmp_loss(
    model: PlayLMP, batch: EpisodeBatch, key: jax.Array, beta: float = 0.5
) -> Float[Array, ""]:
    raise NotImplementedError
    # plans = jax.vmap(model)(
    #     batch.rgb_observations, batch.proprio_observations, batch.episode_lengths
    # )
    # sequence_plans, state_goal_plans = rearrange(
    #     plans, "batch a b d_latent -> a batch b d_latent"
    # )
    # num_sequences = batch.rgb_observations.shape[0]
    # sampling_keys = jax.random.split(key, num_sequences)
    # sampled_plans = jax.vmap(model.sample_plan)(sequence_plans, sampling_keys)
    # goals = batch.rgb_observations[
    #     jnp.arange(num_sequences),
    #     batch.episode_lengths - 1,
    #     ...,
    # ]
    # predicted_actions = jax.vmap(model.policy)(
    #     batch.rgb_observations,
    #     batch.proprio_observations,
    #     goals,
    #     sampled_plans,
    # )
    # reconstruction_loss = jax.vmap(sequence_mse_loss)(
    #     batch.actions, predicted_actions, batch.episode_lengths
    # )
    # plan_kl_loss = jax.vmap(kl_div_diagonal_gaussians)(state_goal_plans, sequence_plans)
    # loss = reconstruction_loss + beta * plan_kl_loss
    # return jnp.mean(loss)


def play_gcbc_loss(model: PlayLMP, batch: EpisodeBatch) -> Float[Array, ""]:
    def instance_loss(
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        actions: Float[Array, "time d_action"],
        episode_length: Int[Array, ""],
    ) -> Float[Array, ""]:
        rgb_goal = rgb_observations[episode_length - 1]
        action_log_likelihoods = model.policy(
            rgb_observations,
            proprio_observations,
            rgb_goal,
            actions,
            jnp.zeros(model.plan_proposal.d_latent),
        )
        return -jnp.mean(
            action_log_likelihoods,
            where=jnp.arange(rgb_observations.shape[0]) < episode_length,
        )

    batch_losses = jax.vmap(instance_loss)(
        batch.rgb_observations,
        batch.proprio_observations,
        batch.actions,
        batch.episode_lengths,
    )
    return jnp.mean(batch_losses)


def kl_div_diagonal_gaussians(
    gaussian_params_p: Float[Array, "2 d_latent"],
    gaussian_params_q: Float[Array, "2 d_latent"],
) -> Float[Array, ""]:
    mean_p, stddev_p = gaussian_params_p
    mean_q, stddev_q = gaussian_params_q
    var_p, var_q = jnp.square(stddev_p), jnp.square(stddev_q)
    return 0.5 * jnp.sum(
        (var_p + jnp.square(mean_p - mean_q)) / var_q + jnp.log(var_q / var_p) - 1
    )
