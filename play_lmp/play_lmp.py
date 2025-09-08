import abc
import dataclasses
from typing import Literal

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree


class AbstractPlanRecognitionNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        actions: Float[Array, "time d_action"],
        sequence_length: Int[Array, ""],
    ) -> distrax.Distribution:
        raise NotImplementedError


class AbstractPlanProposalNetwork(eqx.Module):
    d_latent: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_obs"],
    ) -> distrax.Distribution:
        raise NotImplementedError


class AbstractPolicyNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        actions: Float[Array, "time d_action"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, " time"]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> PyTree:
        raise NotImplementedError

    @abc.abstractmethod
    def act(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
        key: jax.Array,
        state: PyTree,
    ) -> tuple[Float[Array, " d_action"], PyTree]:
        raise NotImplementedError


class PlayLMP(eqx.Module):
    plan_recognizer: AbstractPlanRecognitionNetwork
    plan_proposal: AbstractPlanProposalNetwork
    policy: AbstractPolicyNetwork

    def __init__(
        self,
        plan_recognizer: AbstractPlanRecognitionNetwork,
        plan_proposal: AbstractPlanProposalNetwork,
        policy: AbstractPolicyNetwork,
    ):
        self.plan_recognizer = plan_recognizer
        self.plan_proposal = plan_proposal
        self.policy = policy

    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        actions: Float[Array, "time d_action"],
        sequence_length: Int[Array, ""],
    ) -> tuple[distrax.Distribution, distrax.Distribution]:
        sequence_plan = self.plan_recognizer(observations, actions, sequence_length)
        state_goal_plan = self.plan_proposal(observations[0], goal)
        return sequence_plan, state_goal_plan

    def sample_plan(
        self, distrib: distrax.Distribution, key: jax.Array
    ) -> Float[Array, " d_latent"]:
        return distrib.sample(seed=key)


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
        loss_value, grads = eqx.filter_value_and_grad(_play_gcbc_loss)(model, batch)
    elif method == "play-lmp":
        loss_value, reconstruction_loss, kl_loss = _play_lmp_loss(
            model, batch, key, beta
        )
        grads = eqx.filter_grad(
            lambda model, batch, key, beta: _play_lmp_loss(model, batch, key, beta)[0]
        )(model, batch, key, beta)
        stats = {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, stats


def eval_loss(
    model: PlayLMP,
    mp_policy: jmp.Policy,
    batch: EpisodeBatch,
    key: jax.Array,
    method: Literal["play-lmp", "play-gcbc"],
    beta: float,
) -> tuple[Float[Array, ""], dict]:
    model, batch = mp_policy.cast_to_compute((model, batch))
    stats = {}
    if method == "play-gcbc":
        loss_value = _play_gcbc_loss(model, batch)
    elif method == "play-lmp":
        loss_value, reconstruction_loss, kl_loss = _play_lmp_loss(
            model, batch, key, beta
        )
        stats = {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
    return loss_value, stats


def _play_lmp_loss(
    model: PlayLMP, batch: EpisodeBatch, key: jax.Array, beta: float
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def instance_loss(
        observations: Float[Array, "time d_obs"],
        actions: Float[Array, "time d_action"],
        episode_length: Int[Array, ""],
        key: jax.Array,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        goal = observations[episode_length - 1]
        posterior_plan_distrib, prior_plan_distrib = model(
            observations, goal, actions, episode_length
        )
        sampled_plan = model.sample_plan(posterior_plan_distrib, key)
        action_log_likelihoods = model.policy(observations, goal, actions, sampled_plan)
        action_reconstruction_loss = -jnp.mean(
            action_log_likelihoods,
            where=jnp.arange(observations.shape[0]) < episode_length,
        )
        kl_loss = jnp.asarray(posterior_plan_distrib.kl_divergence(prior_plan_distrib))
        return action_reconstruction_loss, kl_loss

    action_reconstruction_losses, kl_losses = jax.vmap(instance_loss)(
        batch.observations,
        batch.actions,
        batch.episode_lengths,
        jax.random.split(key, batch.observations.shape[0]),
    )
    loss = jnp.mean(action_reconstruction_losses + beta * kl_losses)
    return loss, jnp.mean(action_reconstruction_losses), jnp.mean(kl_losses)


def _play_gcbc_loss(model: PlayLMP, batch: EpisodeBatch) -> Float[Array, ""]:
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
