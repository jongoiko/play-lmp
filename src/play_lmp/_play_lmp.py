import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int


class AbstractPlanRecognitionNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        sequence_length: Int[Array, ""],
    ) -> Float[Array, "2 d_latent"]:
        raise NotImplementedError


class AbstractPlanProposalNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        rgb_observation: Float[Array, "height width channel"],
        proprio_observation: Float[Array, " d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
    ) -> Float[Array, "2 d_latent"]:
        raise NotImplementedError


class AbstractPolicyNetwork(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, "time d_action"]:
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
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        sequence_length: Int[Array, ""],
    ) -> Float[Array, "2 2 d_latent"]:
        sequence_plan = self.plan_recognizer(
            rgb_observations, proprio_observations, sequence_length
        )
        state_goal_plan = self.plan_proposal(
            rgb_observations[0],
            proprio_observations[0],
            rgb_observations[sequence_length - 1],
        )
        return jnp.stack([sequence_plan, state_goal_plan])

    def sample_plan(
        self, params: Float[Array, "2 d_latent"], key: jax.Array
    ) -> Float[Array, " d_latent"]:
        eps = jax.random.normal(key, (params.shape[1],))
        sampled = eps * params[1] + params[0]
        return sampled
