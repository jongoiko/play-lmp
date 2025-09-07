import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from .play_lmp import AbstractPlanProposalNetwork
from .play_lmp import AbstractPlanRecognitionNetwork


class BidirectionalLSTMPlanRecognitionNetwork(AbstractPlanRecognitionNetwork):
    forward_cell: eqx.nn.LSTMCell
    backward_cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(
        self,
        d_obs: int,
        d_latent: int,
        d_action: int,
        hidden_size: int,
        key: jax.Array,
    ):
        fw_key, bw_key, linear_key = jax.random.split(key, 3)
        self.forward_cell = eqx.nn.LSTMCell(d_obs + d_action, hidden_size, key=fw_key)
        self.backward_cell = eqx.nn.LSTMCell(d_obs + d_action, hidden_size, key=bw_key)
        self.linear = eqx.nn.Linear(2 * hidden_size, 2 * d_latent, key=linear_key)

    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        actions: Float[Array, "time d_action"],
        sequence_length: Int[Array, ""],
    ) -> distrax.Distribution:
        padded_sequence_length = observations.shape[0]
        input_features = jnp.concat(
            [
                observations,
                actions,
            ],
            axis=1,
        )

        def fw_scan(
            state: tuple[Array, Array], input: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            output = self.forward_cell(input, state)
            return output, output

        def bw_scan(
            state: tuple[Array, Array], input: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            output = self.backward_cell(input, state)
            return output, output

        init_state = (
            jnp.zeros(self.forward_cell.hidden_size),
            jnp.zeros(self.forward_cell.hidden_size),
        )
        _, (fw_hidden_states, _) = jax.lax.scan(fw_scan, init_state, input_features)
        bw_input_features = jnp.roll(
            input_features[::-1], sequence_length - padded_sequence_length, axis=0
        )
        _, (bw_hidden_states, _) = jax.lax.scan(bw_scan, init_state, bw_input_features)
        bw_hidden_states = jnp.roll(
            bw_hidden_states[::-1], sequence_length - padded_sequence_length, axis=0
        )
        feature_vector = jnp.mean(
            jnp.concat([bw_hidden_states, fw_hidden_states], axis=1),
            axis=0,
            where=jnp.arange(padded_sequence_length).reshape(-1, 1) < sequence_length,
        )
        mean, stddev = rearrange(
            self.linear(feature_vector), "(x d_latent) -> x d_latent", x=2
        )
        stddev = jax.nn.softplus(stddev) + 1e-8
        return distrax.MultivariateNormalDiag(mean, stddev)


class MLPPlanProposalNetwork(AbstractPlanProposalNetwork):
    mlp: eqx.nn.MLP
    d_latent: int

    def __init__(
        self,
        d_obs: int,
        d_latent: int,
        width_size: int,
        depth: int,
        key: jax.Array,
    ):
        self.d_latent = d_latent
        self.mlp = eqx.nn.MLP(2 * d_obs, 2 * d_latent, width_size, depth, key=key)

    def __call__(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_obs"],
    ) -> distrax.Distribution:
        mean, stddev = rearrange(
            self.mlp(jnp.concat([observation, goal])), "(x d_latent) -> x d_latent", x=2
        )
        stddev = jax.nn.softplus(stddev) + 1e-8
        return distrax.MultivariateNormalDiag(mean, stddev)
