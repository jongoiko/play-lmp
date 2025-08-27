from jaxtyping import Array
from jaxtyping import Float


def preprocess_observation(
    obs: Float[Array, " d_obs"],
    mean: Float[Array, " d_obs"],
    std: Float[Array, " d_obs"],
) -> Float[Array, " d_obs"]:
    return (obs - mean) / std


def preprocess_action(
    action: Float[Array, " d_action"],
    max: Float[Array, " d_action"],
    min: Float[Array, " d_action"],
    target_max: Float[Array, " d_action"],
    target_min: Float[Array, " d_action"],
) -> Float[Array, " d_action"]:
    zero_one = (action - min) / (max - min)
    return zero_one * (target_max - target_min) + target_min


def postprocess_action(
    action: Float[Array, " d_action"],
    max: Float[Array, " d_action"],
    min: Float[Array, " d_action"],
    target_max: Float[Array, " d_action"],
    target_min: Float[Array, " d_action"],
) -> Float[Array, " d_action"]:
    zero_one = (action - target_min) / (target_max - target_min)
    return zero_one * (max - min) + min
