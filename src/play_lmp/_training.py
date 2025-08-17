import dataclasses
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

from ._play_lmp import PlayLMP


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EpisodeBatch:
    observations: Float[Array, "batch time d_obs"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]


def make_train_step(
    model: PlayLMP,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    mp_policy: jmp.Policy,
    batch: EpisodeBatch,
    key: jax.Array,
    method: Literal["play-lmp", "play-gcbc"],
    beta: float,
) -> tuple[PlayLMP, PyTree, Float[Array, ""], dict]:
    model, batch = mp_policy.cast_to_compute((model, batch))
    stats = {}
    if method == "play-gcbc":
        loss_value, grads = eqx.filter_value_and_grad(play_gcbc_loss)(model, batch)
    elif method == "play-lmp":
        loss_value, reconstruction_loss, kl_loss = play_lmp_loss(
            model, batch, key, beta
        )
        grads = eqx.filter_grad(
            lambda model, batch, key, beta: play_lmp_loss(model, batch, key, beta)[0]
        )(model, batch, key, beta)
        stats = {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, stats


def play_lmp_loss(
    model: PlayLMP, batch: EpisodeBatch, key: jax.Array, beta: float
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def instance_loss(
        observations: Float[Array, "time d_obs"],
        actions: Float[Array, "time d_action"],
        episode_length: Int[Array, ""],
        key: jax.Array,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        goal = observations[episode_length - 1]
        posterior_plan_params, prior_plan_params = model(
            observations, goal, actions, episode_length
        )
        sampled_plan = model.sample_plan(posterior_plan_params, key)
        action_log_likelihoods = model.policy(observations, goal, actions, sampled_plan)
        action_reconstruction_loss = -jnp.mean(
            action_log_likelihoods,
            where=jnp.arange(observations.shape[0]) < episode_length,
        )
        kl_loss = kl_div_diagonal_gaussians(posterior_plan_params, prior_plan_params)
        return action_reconstruction_loss, kl_loss

    action_reconstruction_losses, kl_losses = jax.vmap(instance_loss)(
        batch.observations,
        batch.actions,
        batch.episode_lengths,
        jax.random.split(key, batch.observations.shape[0]),
    )
    loss = jnp.mean(action_reconstruction_losses + beta * kl_losses)
    return loss, jnp.mean(action_reconstruction_losses), jnp.mean(kl_losses)


def play_gcbc_loss(model: PlayLMP, batch: EpisodeBatch) -> Float[Array, ""]:
    def instance_loss(
        observations: Float[Array, "time d_obs"],
        actions: Float[Array, "time d_action"],
        episode_length: Int[Array, ""],
    ) -> Float[Array, ""]:
        goal = observations[episode_length - 1]
        action_log_likelihoods = model.policy(
            observations,
            goal,
            actions,
            jnp.zeros(model.plan_proposal.d_latent),
        )
        return -jnp.sum(
            action_log_likelihoods,
            where=jnp.arange(observations.shape[0]) < episode_length,
        )

    batch_losses = jax.vmap(instance_loss)(
        batch.observations,
        batch.actions,
        batch.episode_lengths,
    )
    return jnp.sum(batch_losses) / jnp.sum(batch.episode_lengths)


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
