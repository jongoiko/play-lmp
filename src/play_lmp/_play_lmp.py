import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree


class AbstractPlanRecognitionNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        sequence_length: Int[Array, ""],
    ) -> Float[Array, "2 d_latent"]:
        raise NotImplementedError


class AbstractPlanProposalNetwork(eqx.Module):
    d_latent: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_goal"],
    ) -> Float[Array, "2 d_latent"]:
        raise NotImplementedError


class AbstractPolicyNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_goal"],
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
        goal: Float[Array, " d_goal"],
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
        sequence_length: Int[Array, ""],
    ) -> Float[Array, "2 2 d_latent"]:
        sequence_plan = self.plan_recognizer(observations, sequence_length)
        state_goal_plan = self.plan_proposal(
            observations[0], observations[sequence_length - 1]
        )
        return jnp.stack([sequence_plan, state_goal_plan])

    def sample_plan(
        self, params: Float[Array, "2 d_latent"], key: jax.Array
    ) -> Float[Array, " d_latent"]:
        eps = jax.random.normal(key, (params.shape[1],))
        sampled = eps * params[1] + params[0]
        return sampled
