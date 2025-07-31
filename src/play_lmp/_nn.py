import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from einops import repeat
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
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


class CNNEncoder(eqx.Module):
    net: eqx.nn.Sequential
    features_dim: int

    def __init__(self, final_dim: int, key: jax.Array):
        keys = jax.random.split(key, 4)
        self.features_dim = final_dim
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(3, 32, 8, 4, key=keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv2d(32, 32, 3, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.MaxPool2d(2, 2),
                eqx.nn.Conv2d(32, 32, 3, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.MaxPool2d(2, 2),
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(1152, 512, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(512, final_dim, key=keys[3]),
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
    net: eqx.nn.Sequential
    d_latent: int

    def __init__(self, d_proprio: int, d_latent: int, cnn: CNNEncoder, key: jax.Array):
        self.cnn = cnn
        keys = jax.random.split(key, 4)
        self.d_latent = d_latent
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(d_proprio + 2 * cnn.features_dim, 2048, key=keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(2048, 2048, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(2048, 512, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(512, 2 * d_latent, key=keys[3]),
            ]
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
            self.net(input_features), "(x d_latent) -> x d_latent", x=2
        )
        stddev = jax.nn.softplus(stddev)
        return jnp.stack([mean, stddev])


class MLPPolicyNetwork(AbstractPolicyNetwork):
    cnn: CNNEncoder
    net: eqx.nn.Sequential

    def __init__(
        self,
        d_proprio: int,
        d_latent_plan: int,
        d_action: int,
        cnn: CNNEncoder,
        key: jax.Array,
    ):
        self.cnn = cnn
        keys = jax.random.split(key, 4)
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(
                    d_proprio + 2 * cnn.features_dim + d_latent_plan, 2048, key=keys[0]
                ),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(2048, 2048, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(2048, 512, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(512, d_action, key=keys[3]),
            ]
        )

    def __call__(
        self,
        rgb_observations: Float[Array, "time height width channel"],
        proprio_observations: Float[Array, "time d_proprio"],
        rgb_goal: Float[Array, "height width channel"],
        plan: Float[Array, " d_latent"],
    ) -> Float[Array, "time d_action"]:
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
        return jax.vmap(self.net)(input_features)
