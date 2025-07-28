import dataclasses

import jax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Int8


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EpisodeBatch:
    rgb_observations: Int8[Array, "batch time height width channel"]
    proprio_observations: Float[Array, "batch time d_proprio"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]
