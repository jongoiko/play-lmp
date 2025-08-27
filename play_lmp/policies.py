import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from einops import repeat
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PyTree

from .play_lmp import AbstractPolicyNetwork


class MLPPolicyNetwork(AbstractPolicyNetwork):
    mlp: eqx.nn.MLP
    num_dl_mixture_elements: int
    action_max_bound: Float[Array, " d_action"]
    action_min_bound: Float[Array, " d_action"]
    num_action_bins: int

    def __init__(
        self,
        d_obs: int,
        d_latent_plan: int,
        d_action: int,
        width_size: int,
        depth: int,
        num_dl_mixture_elements: int,
        action_max_bound: Float[Array, " d_action"],
        action_min_bound: Float[Array, " d_action"],
        num_action_bins: int,
        key: jax.Array,
    ):
        self.mlp = eqx.nn.MLP(
            2 * d_obs + d_latent_plan,
            3 * num_dl_mixture_elements * d_action,
            width_size,
            depth,
            key=key,
        )
        self.num_dl_mixture_elements = num_dl_mixture_elements
        self.action_max_bound = action_max_bound
        self.action_min_bound = action_min_bound
        self.num_action_bins = num_action_bins

    def _get_dlml_parameters(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
    ) -> tuple[
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
    ]:
        sequence_length = observations.shape[0]
        input_features = jnp.concat(
            [
                observations,
                repeat(goal, "... -> n ...", n=sequence_length),
                repeat(plan, "... -> n ...", n=sequence_length),
            ],
            axis=1,
        )
        means, log_scales, logit_probs = rearrange(
            jax.vmap(self.mlp)(input_features),
            "time (n k d) -> n time d k",
            n=3,
            k=self.num_dl_mixture_elements,
        )
        return means, log_scales, logit_probs

    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        actions: Float[Array, "time d_action"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, " time"]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            observations, goal, plan
        )
        log_likelihoods = jax.vmap(
            _dlml_log_likelihood, in_axes=(0, 0, 0, 0, None, None, None)
        )(
            means,
            log_scales,
            logit_probs,
            actions,
            jax.lax.stop_gradient(self.action_max_bound),
            jax.lax.stop_gradient(self.action_min_bound),
            self.num_action_bins,
        )
        return log_likelihoods

    def reset(self) -> PyTree:
        return None

    def act(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
        key: jax.Array,
        state: PyTree,
    ) -> tuple[Float[Array, " d_action"], PyTree]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            jnp.expand_dims(observation, 0),
            goal,
            plan,
        )
        sample = _dlml_sample(means[0], log_scales[0], logit_probs[0], key)
        return sample, None


class LSTMPolicyNetwork(AbstractPolicyNetwork):
    cell: eqx.nn.LSTMCell
    mlp: eqx.nn.Sequential
    num_dl_mixture_elements: int
    action_max_bound: Float[Array, " d_action"]
    action_min_bound: Float[Array, " d_action"]
    num_action_bins: int

    def __init__(
        self,
        d_obs: int,
        d_latent_plan: int,
        d_action: int,
        hidden_size: int,
        num_dl_mixture_elements: int,
        action_max_bound: Float[Array, " d_action"],
        action_min_bound: Float[Array, " d_action"],
        num_action_bins: int,
        key: jax.Array,
    ):
        lstm_key, mlp_key = jax.random.split(key)
        self.cell = eqx.nn.LSTMCell(
            2 * d_obs + d_latent_plan, hidden_size, key=lstm_key
        )
        mlp_keys = jax.random.split(mlp_key, 2)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(self.cell.hidden_size, 512, key=mlp_keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(
                    512, 3 * num_dl_mixture_elements * d_action, key=mlp_keys[1]
                ),
            ]
        )
        self.num_dl_mixture_elements = num_dl_mixture_elements
        self.action_max_bound = action_max_bound
        self.action_min_bound = action_min_bound
        self.num_action_bins = num_action_bins

    def _get_lstm_input_features(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, "time d"]:
        sequence_length = observations.shape[0]
        features = jnp.concat(
            [
                observations,
                repeat(goal, "... -> n ...", n=sequence_length),
                repeat(plan, "... -> n ...", n=sequence_length),
            ],
            axis=1,
        )
        return features

    def _get_dlml_parameters(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
    ) -> tuple[
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
    ]:
        def lstm_scan(
            state: tuple[Array, Array], input: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            output = self.cell(input, state)
            return output, output

        input_features = self._get_lstm_input_features(
            observations,
            goal,
            plan,
        )
        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )
        _, (hidden_states, _) = jax.lax.scan(lstm_scan, init_state, input_features)
        means, log_scales, logit_probs = rearrange(
            jax.vmap(self.mlp)(hidden_states),
            "time (n k d_action) -> n time d_action k",
            n=3,
            k=self.num_dl_mixture_elements,
        )
        return means, log_scales, logit_probs

    def __call__(
        self,
        observations: Float[Array, "time d_obs"],
        goal: Float[Array, " d_obs"],
        actions: Float[Array, "time d_action"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, " time"]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            observations, goal, plan
        )
        log_likelihoods = jax.vmap(
            _dlml_log_likelihood, in_axes=(0, 0, 0, 0, None, None, None)
        )(
            means,
            log_scales,
            logit_probs,
            actions,
            jax.lax.stop_gradient(self.action_max_bound),
            jax.lax.stop_gradient(self.action_min_bound),
            self.num_action_bins,
        )
        return log_likelihoods

    def reset(self) -> PyTree:
        return (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

    def act(
        self,
        observation: Float[Array, " d_obs"],
        goal: Float[Array, " d_obs"],
        plan: Float[Array, " d_latent"],
        key: jax.Array,
        state: PyTree,
    ) -> tuple[Float[Array, " d_action"], PyTree]:
        input_features = self._get_lstm_input_features(
            jnp.expand_dims(observation, 0),
            goal,
            plan,
        )[0]
        new_state = self.cell(input_features, state)
        means, log_scales, logit_probs = rearrange(
            self.mlp(new_state[0]),
            "(n k d_action) -> n d_action k",
            n=3,
            k=self.num_dl_mixture_elements,
        )
        sample = _dlml_sample(means, log_scales, logit_probs, key)
        return sample, new_state


def _dlml_log_likelihood(
    means: Float[Array, "d_target k"],
    log_scales: Float[Array, "d_target k"],
    logit_probs: Float[Array, "d_target k"],
    single_target: Float[Array, " d_target"],
    target_max_bound: Float[Array, " d_target"],
    target_min_bound: Float[Array, " d_target"],
    num_target_bins: int,
    log_scale_min: float = -7.0,
) -> Float[Array, ""]:
    # https://github.com/lukashermann/hulc/blob/main/hulc/models/decoders/logistic_decoder_rnn.py
    def log_sum_exp(x: Array) -> Array:
        m = jnp.max(x, axis=-1)
        m2 = jnp.max(x, axis=-1, keepdims=True)
        return m + jnp.log(jnp.sum(jnp.exp(x - m2), axis=-1))

    log_scales = jnp.clip(log_scales, min=log_scale_min)
    targets = repeat(single_target, "d -> d k", k=means.shape[1])
    centered_targets = targets - means
    inv_stdv = jnp.exp(-log_scales)
    targets_range = repeat(
        (target_max_bound - target_min_bound) / 2.0, "d -> d k", k=means.shape[1]
    )
    plus_in = inv_stdv * (centered_targets + targets_range / (num_target_bins - 1))
    cdf_plus = jax.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_targets - targets_range / (num_target_bins - 1))
    cdf_min = jax.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
    log_one_minus_cdf_min = -jax.nn.softplus(min_in)
    mid_in = inv_stdv * centered_targets
    log_pdf_mid = mid_in - log_scales - 2.0 * jax.nn.softplus(mid_in)
    cdf_delta = cdf_plus - cdf_min
    log_probs = jnp.where(
        targets < repeat(target_min_bound, "d -> d k", k=means.shape[1]) + 1e-3,
        log_cdf_plus,
        jnp.where(
            targets > repeat(target_max_bound, "d -> d k", k=means.shape[1]) - 1e-3,
            log_one_minus_cdf_min,
            jnp.where(
                cdf_delta > 1e-5,
                jnp.log(jnp.clip(cdf_delta, min=1e-12)),
                log_pdf_mid - jnp.log((num_target_bins - 1) / 2),
            ),
        ),
    )
    log_probs = log_probs + jax.nn.log_softmax(logit_probs, axis=-1)
    return jnp.sum(log_sum_exp(log_probs), axis=-1)


def _dlml_sample(
    means: Float[Array, "d_target k"],
    log_scales: Float[Array, "d_target k"],
    logit_probs: Float[Array, "d_target k"],
    key: jax.Array,
) -> Float[Array, " d_target"]:
    # https://github.com/lukashermann/hulc/blob/main/hulc/models/decoders/logistic_decoder_rnn.py
    r1, r2 = 1e-5, 1.0 - 1e-5
    key1, key2 = jax.random.split(key)
    temp = (r1 - r2) * jax.random.uniform(key1, means.shape) + r2
    temp = logit_probs - jnp.log(-jnp.log(temp))
    argmax = jnp.argmax(temp, -1)
    one_hot_embedding_eye = jnp.eye(means.shape[-1])
    dist = one_hot_embedding_eye[argmax]
    selected_log_scales = (dist * log_scales).sum(axis=-1)
    selected_means = (dist * means).sum(axis=-1)
    scales = jnp.exp(selected_log_scales)
    u = (r1 - r2) * jax.random.uniform(key2, selected_means.shape) + r2
    sampled = selected_means + scales * (jnp.log(u) - jnp.log(1.0 - u))
    return sampled
