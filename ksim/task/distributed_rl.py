# ksim/task/distributed_rl.py
"""Defines a distributed task interface for training reinforcement learning agents."""

import jax
import jax.numpy as jnp
from typing import Any, Generic, TypeVar
from abc import ABC
import xax
from dataclasses import dataclass, replace

from ksim.task.rl import (
    RLTask,
    RLConfig,
    RolloutConstants,
    RolloutSharedState,
    RolloutEnvState,
    LoggedTrajectory,
    RLLoopConstants,
    RLLoopCarry,
)
from ksim.task.ppo import PPOConfig
from ksim.debugging import JitLevel
from ksim.types import RewardState, Trajectory
from jaxtyping import Array, PRNGKeyArray, PyTree

import logging
logger = logging.getLogger(__name__)

@dataclass
class DistributedRLConfig(RLConfig):
    """Configuration for distributed reinforcement learning tasks."""
    
    max_devices_per_host: int | None = xax.field(
        value=None,
        help="Maximum number of devices to use per host. If None, use all available devices.",
    )
    
    gradient_sync_period: int = xax.field(
        value=1,
        help="Synchronize gradients across devices every N steps. Use 1 for every step."
    )
    
    verify_state_consistency: bool = xax.field(
        value=True,
        help="Whether to verify that the state is consistent across devices periodically."
    )


Config = TypeVar("Config", bound=DistributedRLConfig)


# Define axis name for pmap
_PMAP_AXIS_NAME = 'device'


class DistributedRLTask(RLTask[Config], Generic[Config], ABC):
    """Base class for distributed reinforcement learning tasks."""
    
    def get_num_local_devices(self) -> int:
        """Get the number of local devices to use for training."""
        local_device_count = jax.local_device_count()
        if self.config.max_devices_per_host is not None:
            return min(local_device_count, self.config.max_devices_per_host)
        return local_device_count
    
    def shard_env_states(self, env_states: RolloutEnvState) -> RolloutEnvState:
        """Shard environment states across devices."""
        # Reshape env_states to have a leading dimension for devices
        num_devices = self.get_num_local_devices()
        num_envs_per_device = self.config.num_envs // num_devices
        
        def reshape_env_state(x: Array) -> Array:
            # Handle case where x might be a scalar or have an empty shape
            if not hasattr(x, 'shape') or x.shape == ():
                return jnp.repeat(x[None], num_devices)
            return x.reshape(num_devices, num_envs_per_device, *x.shape[1:])
        
        return jax.tree.map(reshape_env_state, env_states)
    
    def unshard_env_states(self, env_states: RolloutEnvState) -> RolloutEnvState:
        """Combine environment states from devices."""
        def combine_env_state(x: Array) -> Array:
            # Handle case where x might be a scalar
            if not hasattr(x, 'shape') or len(x.shape) <= 1:
                return x[0]  # Just take the first device's value for scalars
            return x.reshape(self.config.num_envs, *x.shape[2:])
        
        return jax.tree.map(combine_env_state, env_states)
    
    def replicate_shared_state(self, shared_state: RolloutSharedState) -> RolloutSharedState:
        """Replicate shared state across devices."""
        return jax.device_put_replicated(
            shared_state, jax.local_devices()[:self.get_num_local_devices()]
        )
    
    def initialize_rl_training(
        self,
        mj_model: Any,  # Using Any to avoid circular imports
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopConstants, RLLoopCarry, xax.State]:
        # Get base initialization without manual sharding
        constants, carry, state = super().initialize_rl_training(mj_model, rng)
        
        # For distributed training, we don't manually shard here
        # Instead, we let pmap handle the sharding automatically
        # Just replicate the shared state across devices
        replicated_shared_state = self.replicate_shared_state(carry.shared_state)
        
        # Update carry with replicated shared state but keep env_states unsharded
        distributed_carry = replace(
            carry,
            shared_state=replicated_shared_state,
        )
        
        return constants, distributed_carry, state
    
    def synchronize_gradients(self, grads: PyTree) -> PyTree:
        """Synchronize gradients across devices."""
        return jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
    
    def verify_state_consistency(self, x: PyTree) -> bool:
        """Verify that the state is consistent across devices."""
        # Get the fingerprint (sum of all values) of the state
        def get_fingerprint(tree: PyTree) -> Array:
            sums = jax.tree.map(jnp.sum, tree)
            return jax.tree.reduce(lambda x, y: x + y, sums)
        
        fp = get_fingerprint(x)
        min_fp = jax.lax.pmin(fp, axis_name=_PMAP_AXIS_NAME)
        max_fp = jax.lax.pmax(fp, axis_name=_PMAP_AXIS_NAME)
        
        return jnp.allclose(min_fp, max_fp)
    
    def pmapped_update_model(
        self,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """pmap-compatible version of update_model for distributed training."""
        # Will be implemented in the distributed PPO task
        pass