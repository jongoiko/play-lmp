import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from einops import reduce
from einops import repeat
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
from jaxtyping import UInt8

from ._play_lmp import AbstractPlanProposalNetwork
from ._play_lmp import AbstractPlanRecognitionNetwork
from ._play_lmp import AbstractPolicyNetwork


def preprocess_image(
    image: UInt8[Array, "height width channel"],
    target_size: tuple[int, int, int],
    channel_mean: Float[Array, " channel"],
    channel_std: Float[Array, " channel"],
) -> Float[Array, "target_height target_width channel"]:
    normalized = ((image / 255) - channel_mean) / channel_std
    resized = jax.image.resize(normalized, target_size, "linear")
    return resized


def preprocess_proprio(
    proprio: Float[Array, " d_proprio"],
    mean: Float[Array, " d_proprio"],
    std: Float[Array, " d_proprio"],
) -> Float[Array, " d_proprio"]:
    return (proprio - mean) / std


def preprocess_action(
    action: Float[Array, " d_action"],
    max: Float[Array, " d_action"],
    min: Float[Array, " d_action"],
    target_max: Float[Array, " d_action"],
    target_min: Float[Array, " d_action"],
) -> Float[Array, " d_action"]:
    zero_one = (action - min) / (max - min)
    return zero_one * (target_max - target_min) + target_min


class CNNEncoder(eqx.Module):
    net: eqx.nn.Sequential
    features_dim: int

    def __init__(self, final_dim: int, key: jax.Array):
        keys = jax.random.split(key, 4)
        self.features_dim = final_dim
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(3, 32, 3, key=keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.MaxPool2d(2, 2),
                eqx.nn.Conv2d(32, 64, 3, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.MaxPool2d(4, 4),
                eqx.nn.Conv2d(64, 128, 3, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Lambda(
                    lambda features: reduce(
                        features, "channel height width -> channel", "mean"
                    )
                ),
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(128, final_dim, key=keys[3]),
            ]
        )

    def __call__(
        self, image: Float[Array, "128 128 3"]
    ) -> Float[Array, " features_dim"]:
        return self.net(
            rearrange(image, "height width channel -> channel height width")
        )


class FeedForwardNetwork(eqx.Module):
    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear
    linear_3: eqx.nn.Linear

    def __init__(
        self,
        input_output_dim: int,
        feed_forward_dim: int,
        key: jax.Array,
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        self.linear_1 = eqx.nn.Linear(
            in_features=input_output_dim,
            out_features=feed_forward_dim,
            use_bias=False,
            key=key1,
        )
        self.linear_2 = eqx.nn.Linear(
            in_features=feed_forward_dim,
            out_features=input_output_dim,
            use_bias=False,
            key=key2,
        )
        self.linear_3 = eqx.nn.Linear(
            in_features=input_output_dim,
            out_features=feed_forward_dim,
            use_bias=False,
            key=key3,
        )

    def __call__(self, x: Float[Array, " hidden"]) -> Float[Array, " hidden"]:
        glu = jax.nn.swish(self.linear_1(x)) * self.linear_3(x)
        return self.linear_2(glu)


class TransformerBlock(eqx.Module):
    pre_ff_norm: eqx.nn.RMSNorm
    pre_mha_norm: eqx.nn.RMSNorm
    mha: eqx.nn.MultiheadAttention
    ff: FeedForwardNetwork
    rope: eqx.nn.RotaryPositionalEmbedding

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        rope: eqx.nn.RotaryPositionalEmbedding,
        key: jax.Array,
    ):
        mha_key, ff_key = jax.random.split(key, 2)
        self.pre_ff_norm = eqx.nn.RMSNorm(d_model, use_bias=False)
        self.pre_mha_norm = eqx.nn.RMSNorm(d_model, use_bias=False)
        self.mha = eqx.nn.MultiheadAttention(num_heads, d_model, key=mha_key)
        self.ff = FeedForwardNetwork(d_model, ff_dim, ff_key)
        self.rope = rope

    def __call__(
        self, x: Float[Array, "sequence d_model"], sequence_length: Int[Array, ""]
    ) -> Float[Array, "sequence d_model"]:
        def process_heads(
            query_heads: Float[Array, "sequence num_heads qk_size"],
            key_heads: Float[Array, "sequence num_heads qk_size"],
            value_heads: Float[Array, "sequence num_heads vo_size"],
        ) -> tuple[
            Float[Array, "sequence num_heads qk_size"],
            Float[Array, "sequence num_heads qk_size"],
            Float[Array, "sequence num_heads vo_size"],
        ]:
            query_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(key_heads)
            return query_heads, key_heads, value_heads

        seq_indices = jnp.arange(x.shape[0])
        attn_mask = jnp.logical_and(
            seq_indices < sequence_length,
            seq_indices.reshape(-1, 1) < sequence_length,
        )
        qkv = jax.vmap(self.pre_mha_norm)(x)
        x = x + self.mha(
            qkv, qkv, qkv, mask=attn_mask.astype(jnp.bool), process_heads=process_heads
        )
        x = x + jax.vmap(self.ff)(jax.vmap(self.pre_ff_norm)(x))
        return x


class PlanRecognitionTransformer(AbstractPlanRecognitionNetwork):
    transformer_blocks: list[TransformerBlock]
    cnn: CNNEncoder
    z_linear: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        d_proprio: int,
        num_heads: int,
        ff_dim: int,
        d_latent: int,
        rope_theta: float,
        cnn: CNNEncoder,
        key: jax.Array,
    ):
        d_model = cnn.features_dim + d_proprio
        rope = eqx.nn.RotaryPositionalEmbedding(d_model // num_heads, rope_theta)
        blocks_key, linear_key = jax.random.split(key)
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, ff_dim, rope, key=block_key)
            for block_key in jax.random.split(blocks_key, num_layers)
        ]
        self.cnn = cnn
        self.z_linear = eqx.nn.Linear(d_model, 2 * d_latent, key=linear_key)

    def __call__(
        self,
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        sequence_length: Int[Array, ""],
    ) -> Float[Array, "2 d_latent"]:
        image_features = jax.vmap(self.cnn)(rgb_observations)
        # Proprioceptive features are concatenated to image features at each
        # time step
        tokens = jnp.concat([image_features, proprio_observations], axis=1)
        for block in self.transformer_blocks:
            tokens = block(tokens, sequence_length)
        mask = jnp.arange(tokens.shape[0]).reshape(-1, 1) < sequence_length
        pooled_tokens = jnp.mean(tokens, axis=0, where=mask)
        mean, stddev = rearrange(
            self.z_linear(pooled_tokens), "(x d_latent) -> x d_latent", x=2
        )
        stddev = jax.nn.softplus(stddev)
        return jnp.stack([mean, stddev])


class MLPPlanProposalNetwork(AbstractPlanProposalNetwork):
    cnn: CNNEncoder
    mlp: eqx.nn.MLP
    d_latent: int

    def __init__(
        self,
        d_proprio: int,
        d_latent: int,
        width_size: int,
        depth: int,
        cnn: CNNEncoder,
        key: jax.Array,
    ):
        self.cnn = cnn
        self.d_latent = d_latent
        self.mlp = eqx.nn.MLP(
            d_proprio + 2 * cnn.features_dim, 2 * d_latent, width_size, depth, key=key
        )

    def __call__(
        self,
        rgb_observation: Float[Array, "height width channel"],
        proprio_observation: Float[Array, " d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
    ) -> Float[Array, "2 d_latent"]:
        input_features = jnp.concat(
            [proprio_observation, self.cnn(rgb_observation), self.cnn(rgb_goal)]
        )
        mean, stddev = rearrange(
            self.mlp(input_features), "(x d_latent) -> x d_latent", x=2
        )
        stddev = jax.nn.softplus(stddev)
        return jnp.stack([mean, stddev])


def dlml_log_likelihood(
    means: Float[Array, "d k"],
    log_scales: Float[Array, "d k"],
    logit_probs: Float[Array, "d k"],
    single_target: Float[Array, " d"],
    target_max_bound: Float[Array, " d"],
    target_min_bound: Float[Array, " d"],
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


def dlml_sample(
    means: Float[Array, "d k"],
    log_scales: Float[Array, "d k"],
    logit_probs: Float[Array, "d k"],
    key: jax.Array,
) -> Float[Array, " d"]:
    # https://github.com/lukashermann/hulc/blob/main/hulc/models/decoders/logistic_decoder_rnn.py
    r1, r2 = 1e-5, 1.0 - 1e-5
    key1, key2 = jax.random.split(key)
    temp = (r1 - r2) * jax.random.uniform(key1, means.shape) + r2
    temp = logit_probs - jnp.log(-jnp.log(temp))
    argmax = jnp.argmax(temp, -1)
    selected_log_scales = log_scales[:, argmax].sum(axis=-1)
    selected_means = means[:, argmax].sum(axis=-1)
    scales = jnp.exp(selected_log_scales)
    u = (r1 - r2) * jax.random.uniform(key2, selected_means.shape) + r2
    sampled = selected_means + scales * (jnp.log(u) - jnp.log(1.0 - u))
    return sampled


class MLPPolicyNetwork(AbstractPolicyNetwork):
    cnn: CNNEncoder
    mlp: eqx.nn.MLP
    num_dl_mixture_elements: int
    action_max_bound: Float[Array, " d_action"]
    action_min_bound: Float[Array, " d_action"]
    num_action_bins: int

    def __init__(
        self,
        d_proprio: int,
        d_latent_plan: int,
        d_action: int,
        width_size: int,
        depth: int,
        cnn: CNNEncoder,
        num_dl_mixture_elements: int,
        action_max_bound: Float[Array, " d_action"],
        action_min_bound: Float[Array, " d_action"],
        num_action_bins: int,
        key: jax.Array,
    ):
        self.cnn = cnn
        self.mlp = eqx.nn.MLP(
            d_proprio + 2 * cnn.features_dim + d_latent_plan,
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
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
    ) -> tuple[
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
        Float[Array, "time d_action k"],
    ]:
        image_features = jax.vmap(self.cnn)(rgb_observations)
        sequence_length = rgb_observations.shape[0]
        input_features = jnp.concat(
            [
                image_features,
                proprio_observations,
                repeat(self.cnn(rgb_goal), "... -> n ...", n=sequence_length),
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
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        actions: Float[Array, "time d_action"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, " time"]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            rgb_observations, proprio_observations, rgb_goal, plan
        )
        log_likelihoods = jax.vmap(
            dlml_log_likelihood, in_axes=(0, 0, 0, 0, None, None, None)
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
        rgb_observation: Float[Array, "height width channel"],
        proprio_observation: Float[Array, " d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
        key: jax.Array,
        state: PyTree,
    ) -> tuple[Float[Array, " d_action"], PyTree]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            jnp.expand_dims(rgb_observation, 0),
            jnp.expand_dims(proprio_observation, 0),
            rgb_goal,
            plan,
        )
        sample = dlml_sample(means[0], log_scales[0], logit_probs[0], key)
        return sample, None


class LSTMPolicyNetwork(AbstractPolicyNetwork):
    cnn: CNNEncoder
    cell: eqx.nn.LSTMCell
    mlp: eqx.nn.Sequential
    num_dl_mixture_elements: int
    action_max_bound: Float[Array, " d_action"]
    action_min_bound: Float[Array, " d_action"]
    num_action_bins: int

    def __init__(
        self,
        d_proprio: int,
        d_latent_plan: int,
        d_action: int,
        cnn: CNNEncoder,
        num_dl_mixture_elements: int,
        action_max_bound: Float[Array, " d_action"],
        action_min_bound: Float[Array, " d_action"],
        num_action_bins: int,
        key: jax.Array,
    ):
        self.cnn = cnn
        lstm_key, mlp_key = jax.random.split(key)
        self.cell = eqx.nn.LSTMCell(
            d_proprio + 2 * cnn.features_dim + d_latent_plan, 2048, key=lstm_key
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
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, "time d"]:
        image_features = jax.vmap(self.cnn)(rgb_observations)
        sequence_length = rgb_observations.shape[0]
        features = jnp.concat(
            [
                image_features,
                proprio_observations,
                repeat(self.cnn(rgb_goal), "... -> n ...", n=sequence_length),
                repeat(plan, "... -> n ...", n=sequence_length),
            ],
            axis=1,
        )
        return features

    def _get_dlml_parameters(
        self,
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
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
            rgb_observations,
            proprio_observations,
            rgb_goal,
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
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        actions: Float[Array, "time d_action"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, " time"]:
        means, log_scales, logit_probs = self._get_dlml_parameters(
            rgb_observations, proprio_observations, rgb_goal, plan
        )
        log_likelihoods = jax.vmap(
            dlml_log_likelihood, in_axes=(0, 0, 0, 0, None, None, None)
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
        rgb_observation: Float[Array, "height width channel"],
        proprio_observation: Float[Array, " d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
        key: jax.Array,
        state: PyTree,
    ) -> tuple[Float[Array, " d_action"], PyTree]:
        input_features = self._get_lstm_input_features(
            jnp.expand_dims(rgb_observation, 0),
            jnp.expand_dims(proprio_observation, 0),
            rgb_goal,
            plan,
        )[0]
        new_state = self.cell(input_features, state)
        means, log_scales, logit_probs = rearrange(
            self.mlp(new_state[0]),
            "(n k d_action) -> n d_action k",
            n=3,
            k=self.num_dl_mixture_elements,
        )
        sample = dlml_sample(means, log_scales, logit_probs, key)
        return sample, new_state
