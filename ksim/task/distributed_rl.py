"""Core distributed RL functionality for multi-GPU training."""

__all__ = [
    "DistributedRLConfig", 
    "DistributedRLTask",
    "get_device_info",
    "replicate_state",
    "shard_environments",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask


def get_device_info() -> Tuple[int, int, int]:
    """Gets information about available devices.
    
    Returns:
        Tuple of (process_count, process_id, local_device_count)
    """
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    return process_count, process_id, local_device_count


def replicate_state(state: PyTree, num_devices: int) -> PyTree:
    """Replicates state across devices.
    
    Args:
        state: State to replicate
        num_devices: Number of devices to replicate across
        
    Returns:
        Replicated state with leading device dimension
    """
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape) if hasattr(x, "shape") else x,
        state
    )


def shard_environments(env_state: PyTree, num_devices: int, num_envs_per_device: int) -> PyTree:
    """Shards environment states across devices.
    
    Args:
        env_state: Environment state to shard
        num_devices: Number of devices
        num_envs_per_device: Number of environments per device
        
    Returns:
        Sharded environment state with shape (num_devices, num_envs_per_device, ...)
    """
    def reshape_for_devices(x):
        if hasattr(x, "shape") and len(x.shape) > 0:
            total_envs = x.shape[0]
            expected_total = num_devices * num_envs_per_device
            
            if total_envs != expected_total:
                raise ValueError(
                    f"Total environments ({total_envs}) doesn't match "
                    f"num_devices * num_envs_per_device ({expected_total})"
                )
            
            # Reshape from (total_envs, ...) to (num_devices, num_envs_per_device, ...)
            return x.reshape(num_devices, num_envs_per_device, *x.shape[1:])
        return x
    
    return jax.tree.map(reshape_for_devices, env_state)


@dataclass
class DistributedRLConfig(RLConfig):
    """Configuration for distributed RL training."""
    
    pmap_axis_name: str = xax.field(
        value="i",
        help="Axis name for pmapping operations."
    )
    
    grad_sync_period: int = xax.field(
        value=1,
        help="Synchronize gradients across devices every N updates."
    )


class DistributedRLTask(RLTask, ABC):
    """Base class for distributed reinforcement learning tasks.
    
    Extends RLTask with multi-device training capabilities using jax.pmap.
    """
    
    config: DistributedRLConfig
    
    def __init__(self, config: DistributedRLConfig):
        super().__init__(config)
        self.num_devices = jax.local_device_count()
        
        # Ensure num_envs is divisible by num_devices
        if config.num_envs % self.num_devices != 0:
            raise ValueError(
                f"num_envs ({config.num_envs}) must be divisible by num_devices ({self.num_devices})"
            )
        
        self.num_envs_per_device = config.num_envs // self.num_devices
        
    def get_device_count(self) -> int:
        """Returns the number of devices available for training."""
        return self.num_devices
        
    def get_envs_per_device(self) -> int:
        """Returns the number of environments per device."""
        return self.num_envs_per_device
    
    @abstractmethod
    def distributed_rollout(
        self,
        model: PyTree,
        rollout_constants: PyTree,
        rollout_shared_state: PyTree,
        rollout_env_state: PyTree,
        rng: PRNGKeyArray,
    ) -> Tuple[PyTree, PyTree, PyTree]:
        """Performs distributed rollout across devices.
        
        Args:
            model: Model parameters (replicated across devices)
            rollout_constants: Constants for rollout (replicated)
            rollout_shared_state: Shared state (replicated)
            rollout_env_state: Environment state (sharded across devices)
            rng: Random key (different per device)
            
        Returns:
            Tuple of (trajectories, updated_env_state, updated_shared_state)
        """
        pass
        
    @abstractmethod
    def distributed_update_model(
        self,
        model: PyTree,
        trajectories: PyTree,
        rewards: PyTree,
        rng: PRNGKeyArray,
    ) -> Tuple[PyTree, PyTree]:
        """Performs distributed model update with gradient synchronization.
        
        Args:
            model: Current model parameters (replicated)
            trajectories: Training trajectories (sharded across devices)
            rewards: Computed rewards (sharded across devices)
            rng: Random key for updates
            
        Returns:
            Tuple of (updated_model, metrics)
        """
        pass

    def run_distributed_training(self) -> None:
        """Main training loop for distributed RL."""
        print(f"Starting distributed training on {self.num_devices} devices")
        print(f"Environments per device: {self.num_envs_per_device}")
        
        # Initialize distributed training state
        model = self.get_model(jax.random.PRNGKey(0))
        rollout_constants = self._get_rollout_constants()
        
        # Replicate model and constants across devices
        replicated_model = replicate_state(model, self.num_devices)
        replicated_constants = replicate_state(rollout_constants, self.num_devices)
        
        # Initialize and shard environment state
        initial_env_state = self._get_initial_env_state()
        sharded_env_state = shard_environments(
            initial_env_state, self.num_devices, self.num_envs_per_device
        )
        
        # Initialize shared state (replicated)
        initial_shared_state = self._get_initial_shared_state()
        replicated_shared_state = replicate_state(initial_shared_state, self.num_devices)
        
        # Create pmapped functions
        pmapped_rollout = jax.pmap(
            self.distributed_rollout,
            axis_name=self.config.pmap_axis_name,
        )
        
        pmapped_update = jax.pmap(
            self.distributed_update_model,
            axis_name=self.config.pmap_axis_name,
        )
        
        # Training loop
        rng = jax.random.PRNGKey(42)
        for step in range(1000):  # TODO: Configure number of steps
            rng, rollout_rng, update_rng = jax.random.split(rng, 3)
            
            # Split RNG for each device
            device_rngs = jax.random.split(rollout_rng, self.num_devices)
            
            # Perform distributed rollout
            trajectories, sharded_env_state, replicated_shared_state = pmapped_rollout(
                replicated_model,
                replicated_constants,
                replicated_shared_state,
                sharded_env_state,
                device_rngs,
            )
            
            # Compute rewards (this could also be pmapped if needed)
            rewards = self._compute_rewards(trajectories)
            
            # Perform distributed model update
            if step % self.config.grad_sync_period == 0:
                device_update_rngs = jax.random.split(update_rng, self.num_devices)
                replicated_model, metrics = pmapped_update(
                    replicated_model,
                    trajectories,
                    rewards,
                    device_update_rngs,
                )
                
                if step % 100 == 0:
                    print(f"Step {step}, metrics: {metrics}")
    
    def _get_rollout_constants(self) -> PyTree:
        """Gets constants needed for rollout. Override in subclasses."""
        return {}
    
    def _get_initial_env_state(self) -> PyTree:
        """Gets initial environment state. Override in subclasses.""" 
        return {}
    
    def _get_initial_shared_state(self) -> PyTree:
        """Gets initial shared state. Override in subclasses."""
        return {}
    
    def _compute_rewards(self, trajectories: PyTree) -> PyTree:
        """Computes rewards from trajectories. Override in subclasses."""
        return {}