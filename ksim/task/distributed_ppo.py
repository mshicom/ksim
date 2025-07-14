"""Distributed PPO implementation for multi-GPU training."""

__all__ = [
    "DistributedPPOConfig",
    "DistributedPPOTask",
]

from abc import ABC
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.distributed_rl import DistributedRLConfig, DistributedRLTask
from ksim.task.ppo import PPOConfig, compute_ppo_inputs, compute_ppo_loss


@dataclass
class DistributedPPOConfig(DistributedRLConfig, PPOConfig):
    """Configuration for distributed PPO training."""
    pass


class DistributedPPOTask(DistributedRLTask, ABC):
    """Distributed PPO task with multi-device training capabilities."""
    
    config: DistributedPPOConfig
    
    def distributed_rollout(
        self,
        model: PyTree,
        rollout_constants: PyTree,
        rollout_shared_state: PyTree,
        rollout_env_state: PyTree,
        rng: PRNGKeyArray,
    ) -> Tuple[PyTree, PyTree, PyTree]:
        """Performs distributed rollout using pmapped operations.
        
        This function runs on each device and processes a subset of environments.
        
        Args:
            model: Model parameters for this device
            rollout_constants: Rollout constants for this device
            rollout_shared_state: Shared state for this device
            rollout_env_state: Environment state for this device's environments
            rng: Random key for this device
            
        Returns:
            Tuple of (trajectories, updated_env_state, updated_shared_state)
        """
        # This would typically call the single-device rollout function
        # but operating on the device's subset of environments
        
        # For now, return placeholder - this would be implemented to call
        # the actual rollout logic from the parent PPOTask
        trajectories = self._single_device_rollout(
            model, rollout_constants, rollout_shared_state, rollout_env_state, rng
        )
        
        return trajectories, rollout_env_state, rollout_shared_state
    
    def distributed_update_model(
        self,
        model: PyTree,
        trajectories: PyTree,
        rewards: PyTree,
        rng: PRNGKeyArray,
    ) -> Tuple[PyTree, PyTree]:
        """Performs distributed model update with gradient synchronization.
        
        This function:
        1. Computes PPO variables locally on each device
        2. Computes gradients locally 
        3. Synchronizes gradients across devices using pmean
        4. Applies gradients to update the model
        
        Args:
            model: Current model parameters for this device
            trajectories: Trajectories from this device's environments
            rewards: Rewards from this device's environments  
            rng: Random key for this device
            
        Returns:
            Tuple of (updated_model, metrics)
        """
        # Step 1: Compute PPO inputs (advantages, value targets, etc.)
        ppo_inputs = self._compute_ppo_inputs_for_device(trajectories, rewards)
        
        # Step 2: Get on-policy variables (baseline for comparison)
        on_policy_variables = self._get_on_policy_variables(model, trajectories)
        
        # Step 3: Define function to compute loss and gradients
        def loss_and_grad_fn(model_params):
            # Get off-policy variables with current model
            off_policy_variables = self._get_off_policy_variables(model_params, trajectories)
            
            # Compute PPO loss
            losses = compute_ppo_loss(
                on_policy_variables=on_policy_variables,
                off_policy_variables=off_policy_variables,
                ppo_inputs=ppo_inputs,
                clip_param=self.config.clip_param,
                value_loss_coef=self.config.value_loss_coef,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                use_clipped_value_loss=self.config.use_clipped_value_loss,
                log_clip_value=self.config.log_clip_value,
            )
            
            # Total loss is sum of all loss components
            total_loss = sum(losses.values())
            
            return total_loss, losses
        
        # Step 4: Compute gradients
        (total_loss, loss_components), grads = jax.value_and_grad(
            loss_and_grad_fn, has_aux=True
        )(model)
        
        # Step 5: Synchronize gradients across devices using pmean
        # This is the key step for distributed training
        synced_grads = jax.lax.pmean(grads, axis_name=self.config.pmap_axis_name)
        
        # Step 6: Apply gradients to update model
        optimizer = self._get_optimizer()
        opt_state = self._get_optimizer_state()
        
        updates, new_opt_state = optimizer.update(synced_grads, opt_state, model)
        updated_model = eqx.apply_updates(model, updates)
        
        # Step 7: Compute metrics (also synchronized for consistency)
        metrics = {
            "total_loss": jax.lax.pmean(total_loss, axis_name=self.config.pmap_axis_name),
            **{k: jax.lax.pmean(v.mean(), axis_name=self.config.pmap_axis_name) 
               for k, v in loss_components.items()}
        }
        
        return updated_model, metrics
    
    def _single_device_rollout(
        self,
        model: PyTree,
        rollout_constants: PyTree,
        rollout_shared_state: PyTree,
        rollout_env_state: PyTree,
        rng: PRNGKeyArray,
    ) -> PyTree:
        """Performs rollout on a single device's environments.
        
        This is a placeholder - in a full implementation, this would
        call the existing rollout logic but handle the device-specific
        environment subset.
        """
        # Placeholder trajectory structure
        num_envs_device = self.get_envs_per_device()
        rollout_steps = int(self.config.rollout_length_seconds / self.config.ctrl_dt)
        
        return {
            "observations": jnp.zeros((rollout_steps, num_envs_device, 10)),  # Placeholder obs size
            "actions": jnp.zeros((rollout_steps, num_envs_device, 5)),        # Placeholder action size
            "rewards": jnp.zeros((rollout_steps, num_envs_device)),
            "dones": jnp.zeros((rollout_steps, num_envs_device), dtype=bool),
        }
    
    def _compute_ppo_inputs_for_device(self, trajectories: PyTree, rewards: PyTree) -> PyTree:
        """Computes PPO inputs (advantages, value targets) for this device's data."""
        # Placeholder - would use actual compute_ppo_inputs function
        num_envs_device = self.get_envs_per_device() 
        rollout_steps = int(self.config.rollout_length_seconds / self.config.ctrl_dt)
        
        return {
            "advantages_t": jnp.zeros((rollout_steps, num_envs_device)),
            "value_targets_t": jnp.zeros((rollout_steps, num_envs_device)),
            "gae_t": jnp.zeros((rollout_steps, num_envs_device)),
            "returns_t": jnp.zeros((rollout_steps, num_envs_device)),
        }
    
    def _get_on_policy_variables(self, model: PyTree, trajectories: PyTree) -> PyTree:
        """Gets on-policy variables using the old model."""
        # Placeholder - would compute log_probs, values, etc. with old model
        num_envs_device = self.get_envs_per_device()
        rollout_steps = int(self.config.rollout_length_seconds / self.config.ctrl_dt)
        
        return {
            "log_probs": jnp.zeros((rollout_steps, num_envs_device)),
            "values": jnp.zeros((rollout_steps, num_envs_device)),
            "entropy": jnp.zeros((rollout_steps, num_envs_device)),
        }
    
    def _get_off_policy_variables(self, model: PyTree, trajectories: PyTree) -> PyTree:
        """Gets off-policy variables using the current model."""
        # Placeholder - would compute log_probs, values, etc. with current model
        num_envs_device = self.get_envs_per_device()
        rollout_steps = int(self.config.rollout_length_seconds / self.config.ctrl_dt)
        
        return {
            "log_probs": jnp.zeros((rollout_steps, num_envs_device)),
            "values": jnp.zeros((rollout_steps, num_envs_device)),
            "entropy": jnp.zeros((rollout_steps, num_envs_device)),
        }
    
    def _get_optimizer(self):
        """Gets the optimizer for model updates."""
        # Would return actual optimizer (e.g., Adam)
        import optax
        return optax.adam(learning_rate=self.config.learning_rate)
    
    def _get_optimizer_state(self):
        """Gets the current optimizer state."""
        # Would return actual optimizer state
        return None
    
    def verify_distributed_setup(self) -> None:
        """Verifies that the distributed training setup is correct."""
        print("=== Distributed Training Setup ===")
        print(f"Number of devices: {self.num_devices}")
        print(f"Total environments: {self.config.num_envs}")
        print(f"Environments per device: {self.num_envs_per_device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient sync period: {self.config.grad_sync_period}")
        print(f"PMmap axis name: {self.config.pmap_axis_name}")
        
        # Verify divisibility constraints
        assert self.config.num_envs % self.num_devices == 0, \
            f"num_envs ({self.config.num_envs}) must be divisible by num_devices ({self.num_devices})"
        
        # Verify batch constraints if specified
        if hasattr(self.config, 'num_minibatches'):
            total_samples = self.config.batch_size * self.config.num_minibatches
            assert total_samples <= self.config.num_envs, \
                f"batch_size * num_minibatches ({total_samples}) must be <= num_envs ({self.config.num_envs})"
        
        print("✓ All distributed training constraints satisfied")
        print("=" * 35)