"""Distributed reinforcement learning task implementation using JAX pmap for multi-device training."""

__all__ = [
    "DistributedRLConfig",
    "DistributedRLTask",
    "get_device_info",
    "replicate_state",
    "shard_environments",
]

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PyTree

from ksim.task.rl import RLConfig, RLTask

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class DistributedRLConfig(RLConfig):
    """Configuration for distributed RL training."""

    pmap_axis_name: str = xax.field(
        value="device",
        help="Axis name for pmapping operations."
    )

    grad_sync_period: int = xax.field(
        value=1,
        help="Synchronize gradients across devices every N updates."
    )


Config = TypeVar("Config", bound=DistributedRLConfig)


def get_device_info() -> tuple[int, int, int]:
    """Gets information about available devices.

    Returns:
        Tuple of (process_count, process_id, local_device_count).
    """
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    return process_count, process_id, local_device_count


def replicate_state(state: PyTree) -> PyTree:
    """Replicates state across all devices.

    Args:
        state: The state to replicate.

    Returns:
        The replicated state.
    """
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (jax.local_device_count(),) + x.shape), state)


def shard_environments(
    env_state: PyTree,
    num_devices: int,
    num_envs: int
) -> PyTree:
    """Shards environment states across devices.

    Args:
        env_state: The environment state to shard.
        num_devices: Number of devices to shard across.
        num_envs: Total number of environments.

    Returns:
        The sharded environment state.
    """
    if num_envs % num_devices != 0:
        raise ValueError(f"Number of environments ({num_envs}) must be divisible by number of devices ({num_devices})")

    num_envs_per_device = num_envs // num_devices

    def reshape_for_sharding(x: Array) -> Array:
        if hasattr(x, "shape") and len(x.shape) > 0:
            # Reshape to (num_devices, num_envs_per_device, ...)
            return x.reshape(num_devices, num_envs_per_device, *x.shape[1:])
        return x

    return jax.tree.map(reshape_for_sharding, env_state)


class DistributedRLTask(RLTask[Config], Generic[Config], ABC):
    """Base class for distributed RL tasks using JAX pmap."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Get device information
        self.process_count, self.process_id, self.local_device_count = get_device_info()

        # Validate configuration for distributed training
        self._validate_distributed_config()

        logger.info(
            "Distributed training setup: %d local devices, process %d/%d",
            self.local_device_count,
            self.process_id,
            self.process_count,
        )

    def _validate_distributed_config(self) -> None:
        """Validates the configuration for distributed training."""
        if self.config.num_envs % self.local_device_count != 0:
            raise ValueError(
                f"Number of environments ({self.config.num_envs}) must be divisible by "
                f"number of devices ({self.local_device_count})"
            )

        if self.config.batch_size * self.num_batches != self.config.num_envs:
            raise ValueError(
                f"Batch size ({self.config.batch_size}) times number of batches ({self.num_batches}) "
                f"must equal number of environments ({self.config.num_envs})"
            )

    @property
    def num_envs_per_device(self) -> int:
        """Number of environments per device."""
        return self.config.num_envs // self.local_device_count

    def verify_distributed_setup(self) -> None:
        """Verifies the distributed training setup."""
        logger.info("=== Distributed Training Setup ===")
        logger.info("Total devices: %d", self.local_device_count)
        logger.info("Total environments: %d", self.config.num_envs)
        logger.info("Environments per device: %d", self.num_envs_per_device)
        logger.info("Batch size: %d", self.config.batch_size)
        logger.info("Number of batches: %d", self.num_batches)
        logger.info("Pmap axis name: %s", self.config.pmap_axis_name)
        logger.info("Gradient sync period: %d", self.config.grad_sync_period)
        logger.info("==================================")

        # Additional validation
        if self.local_device_count == 1:
            logger.warning("Only 1 device detected. Distributed training will fall back to single-device mode.")
