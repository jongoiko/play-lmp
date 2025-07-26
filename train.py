from __future__ import annotations

import dataclasses
from typing import cast

import hydra
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Int8
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    dataset = get_dataset(cfg)
    for batch in dataset:
        episode_batch = EpisodeBatch.from_tfds_batch(batch)  # type: ignore
        # TODO: Training loop on episode_batch
        del episode_batch
    del random_key


@dataclasses.dataclass
class EpisodeBatch:
    rgb_observations: Int8[Array, "batch time height width channel"]
    proprio_observations: Float[Array, "batch time d_proprio"]
    actions: Float[Array, "batch time d_action"]
    episode_lengths: Int[Array, " batch"]

    @staticmethod
    def from_tfds_batch(batch: dict) -> EpisodeBatch:
        rgb_observations = jnp.asarray(batch["observation"]["rgb"])
        proprio_observations = jnp.asarray(batch["observation"]["effector_translation"])
        actions = jnp.asarray(batch["action"])
        # The "valid" key is introduced by `pad_to_cardinality`
        episode_lengths = jnp.asarray(batch["valid"]).sum(axis=1)
        return EpisodeBatch(
            rgb_observations=rgb_observations,
            proprio_observations=proprio_observations,
            actions=actions,
            episode_lengths=episode_lengths,
        )


def get_dataset(cfg: DictConfig) -> tf.data.Dataset:
    builder = tfds.builder_from_directory(cfg.training.tf_dataset_path)
    dataset = cast(tf.data.Dataset, builder.as_dataset(split="train"))
    return (
        dataset.repeat()
        .shuffle(buffer_size=cfg.training.shuffle_buffer_size, seed=cfg.random_seed)
        .flat_map(
            lambda x: x["steps"].apply(
                tf.data.experimental.pad_to_cardinality(
                    cfg.training.sequence_padding_length
                )
            ),
        )
        .batch(cfg.training.sequence_padding_length)
        .batch(cfg.training.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


if __name__ == "__main__":
    main()
