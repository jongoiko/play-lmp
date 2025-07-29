import dataclasses

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from ._nn import PlayLMP


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EpisodeBatch:
    rgb_observations: Float[Array, "batch time height width channel"]
    proprio_observations: Float[Array, "batch time d_proprio"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]


def play_lmp_loss(
    model: PlayLMP, batch: EpisodeBatch, key: jax.Array, beta: float = 0.5
) -> Float[Array, ""]:
    plans = jax.vmap(model)(
        batch.rgb_observations, batch.proprio_observations, batch.episode_lengths
    )
    sequence_plans, state_goal_plans = rearrange(
        plans, "batch a b d_latent -> a batch b d_latent"
    )
    num_sequences = batch.rgb_observations.shape[0]
    sampling_keys = jax.random.split(key, num_sequences)
    sampled_plans = jax.vmap(model.sample_plan)(state_goal_plans, sampling_keys)
    predicted_actions = jax.vmap(model.policy)(
        batch.rgb_observations,
        batch.proprio_observations,
        batch.rgb_observations[:, -1, ...],
        sampled_plans,
    )
    reconstruction_loss = jax.vmap(sequence_mse_loss)(
        batch.actions, predicted_actions, batch.episode_lengths
    )
    plan_kl_loss = jax.vmap(kl_div_diagonal_gaussians)(state_goal_plans, sequence_plans)
    loss = reconstruction_loss + beta * plan_kl_loss
    return jnp.mean(loss)


def sequence_mse_loss(
    y_real: Float[Array, "time d"],
    y_pred: Float[Array, "time d"],
    sequence_length: Int[Array, ""],
) -> Float[Array, ""]:
    def mse_loss(
        y_real: Float[Array, " d"],
        y_pred: Float[Array, " d"],
    ) -> Float[Array, ""]:
        return jnp.square(y_pred - y_real).sum()

    errors = jax.vmap(mse_loss)(y_real, y_pred)
    return jnp.mean(errors, where=jnp.arange(y_real.shape[0]) < sequence_length)


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
