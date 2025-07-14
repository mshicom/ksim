"""Distributed multi-GPU training support for KSIM RL tasks."""

__all__ = [
    "DistributedRLTask",
    "DistributedPPOTask",
]

import logging
from dataclasses import replace
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.debugging import JitLevel
from ksim.task.ppo import PPOConfig, PPOTask, PPOVariables
from ksim.task.rl import RLConfig, RLLoopCarry, RLLoopConstants, RLTask
from ksim.types import LoggedTrajectory, Metrics, RewardState, Trajectory

logger = logging.getLogger(__name__)

# Pmap axis name for multi-device parallelism (following Brax convention)
_PMAP_AXIS_NAME = "i"

ConfigType = TypeVar("ConfigType", bound=RLConfig)


def _device_put_sharded(tree: Array, axis: int = 0) -> Array:
    """Put a tree of arrays on devices, sharded along the specified axis."""
    devices = jax.local_devices()
    if len(devices) == 1:
        return tree
    
    def shard_array(arr: Array) -> Array:
        if arr.shape[axis] % len(devices) != 0:
            raise ValueError(
                f"Cannot shard array with shape {arr.shape} along axis {axis} "
                f"across {len(devices)} devices: {arr.shape[axis]} is not divisible by {len(devices)}"
            )
        
        # Split array along the sharding axis
        chunks = jnp.split(arr, len(devices), axis=axis)
        return jax.device_put_sharded(chunks, devices)
    
    return jax.tree.map(shard_array, tree)


def _device_put_replicated(tree: Array) -> Array:
    """Put a tree of arrays on devices, replicated across all devices."""
    devices = jax.local_devices()
    if len(devices) == 1:
        return tree
    return jax.device_put_replicated(tree, devices)


def _reshape_for_devices(tree: Array, num_devices: int) -> Array:
    """Reshape arrays to have a leading dimension for devices."""
    def reshape_array(x: Array) -> Array:
        if x.size == 0:
            # Handle 0-sized arrays safely
            return x.reshape((num_devices, 0) + x.shape[1:])
        
        if x.shape[0] % num_devices != 0:
            raise ValueError(
                f"Cannot reshape array with shape {x.shape} for {num_devices} devices: "
                f"{x.shape[0]} is not divisible by {num_devices}"
            )
        
        num_env_per_device = x.shape[0] // num_devices
        return x.reshape((num_devices, num_env_per_device) + x.shape[1:])
    
    return jax.tree.map(reshape_array, tree)


def _unreplicate_first_device(tree: Array) -> Array:
    """Extract data from the first device, assuming it's replicated."""
    return jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, tree)


class DistributedRLTask(RLTask[ConfigType], Generic[ConfigType]):
    """Distributed multi-GPU version of RLTask."""
    
    @property
    def num_devices(self) -> int:
        """Number of available devices for training."""
        return len(jax.local_devices())
    
    @property
    def use_distributed_training(self) -> bool:
        """Whether to use distributed training."""
        return self.num_devices > 1
    
    def _rl_train_loop_step(
        self,
        carry: RLLoopCarry,
        constants: RLLoopConstants,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, Metrics, LoggedTrajectory]:
        """Distributed version of the RL training loop step."""
        
        if not self.use_distributed_training:
            # Fall back to single-device implementation
            return super()._rl_train_loop_step(carry, constants, state, rng)
        
        # Multi-device implementation using pmap
        jax.debug.print("Running distributed training step on {} devices", self.num_devices)
        
        def single_step_fn(
            carry_i: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, tuple[Metrics, LoggedTrajectory]]:
            # Parallel rollout across devices
            pmapped_unroll = jax.pmap(
                lambda constants, env_states, shared_state: xax.vmap(
                    self._single_unroll,
                    in_axes=(None, 0, None),
                    jit_level=JitLevel.UNROLL,
                )(constants, env_states, shared_state),
                axis_name=_PMAP_AXIS_NAME,
            )
            
            # Reshape states for device distribution
            carry_reshaped = jax.tree.map(
                lambda x: _reshape_for_devices(x, self.num_devices),
                carry_i
            )
            
            trajectories, rewards, env_state = pmapped_unroll(
                constants.constants,
                carry_reshaped.env_states,
                carry_reshaped.shared_state,
            )
            
            # Flatten trajectories and rewards back to single device dimension
            trajectories = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                trajectories
            )
            rewards = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                rewards
            )
            env_state = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                env_state
            )
            
            # Update carry with new env states
            carry_i = carry_i.replace(env_states=env_state)
            
            # Runs update on the previous trajectory
            carry_i, train_metrics, logged_traj = self.update_model(
                constants=constants,
                carry=carry_i,
                trajectories=trajectories,
                rewards=rewards,
                rng=rng,
            )
            
            # Store all the metrics to log
            metrics = Metrics(
                train=train_metrics,
                reward=xax.FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=xax.FrozenDict(self.get_termination_metrics(trajectories)),
                curriculum_level=carry_i.env_states.curriculum_state.level,
            )
            
            # Steps the curriculum
            curriculum_state = constants.constants.curriculum(
                trajectory=trajectories,
                rewards=rewards,
                training_state=state,
                prev_state=carry_i.env_states.curriculum_state,
            )
            
            # Updates the curriculum state
            carry_i = carry_i.replace(
                env_states=carry_i.env_states.replace(curriculum_state=curriculum_state)
            )
            
            return carry_i, (metrics, logged_traj)
        
        for _ in range(self.config.epochs_per_log_step):
            rng, step_rng = jax.random.split(rng)
            carry, (metrics, logged_traj) = single_step_fn(carry, step_rng)
        
        return carry, metrics, logged_traj


class DistributedPPOTask(PPOTask[ConfigType], DistributedRLTask[ConfigType], Generic[ConfigType]):
    """Distributed multi-GPU version of PPOTask."""
    
    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[
        RLLoopCarry,
        xax.FrozenDict[str, Array],
        LoggedTrajectory,
    ]:
        """Distributed version of model update with pmap for gradient computation."""
        
        if not self.use_distributed_training:
            # Fall back to single-device implementation
            return super().update_model(
                constants=constants,
                carry=carry,
                trajectories=trajectories,
                rewards=rewards,
                rng=rng,
            )
        
        jax.debug.print("Running distributed model update on {} devices", self.num_devices)
        
        # Gets the policy model
        policy_model_arr = carry.shared_state.model_arrs[0]
        policy_model_static = constants.constants.model_statics[0]
        policy_model = eqx.combine(policy_model_arr, policy_model_static)
        
        # Runs the policy model on the trajectory to get the PPO variables
        on_policy_rngs = jax.random.split(rng, self.config.num_envs)
        ppo_fn = xax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0), jit_level=JitLevel.RL_CORE)
        on_policy_variables, _ = ppo_fn(policy_model, trajectories, carry.env_states.model_carry, on_policy_rngs)
        on_policy_variables = jax.tree.map(lambda x: jax.lax.stop_gradient(x), on_policy_variables)
        
        # Loops over the trajectory batches and applies gradient updates
        def update_model_in_batch(
            carry: RLLoopCarry,
            xs: tuple[Array, PRNGKeyArray],
        ) -> tuple[RLLoopCarry, tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            batch_indices, rng = xs
            rng, batch_rng = jax.random.split(rng)
            
            # Gets the current batch of trajectories and rewards
            trajectory_batch = jax.tree.map(lambda x: x[batch_indices], trajectories)
            reward_batch = jax.tree.map(lambda x: x[batch_indices], rewards)
            env_states_batch = jax.tree.map(lambda x: x[batch_indices], carry.env_states)
            on_policy_variables_batch = jax.tree.map(lambda x: x[batch_indices], on_policy_variables)
            
            # For distributed training, use pmap for gradient computation
            if len(batch_indices) >= self.num_devices:
                next_carry, metrics, logged_traj = self._distributed_single_step(
                    trajectories=trajectory_batch,
                    rewards=reward_batch,
                    constants=constants,
                    carry=replace(carry, env_states=env_states_batch),
                    on_policy_variables=on_policy_variables_batch,
                    rng=batch_rng,
                )
            else:
                # Fall back to single device for small batches
                next_carry, metrics, logged_traj = self._single_step(
                    trajectories=trajectory_batch,
                    rewards=reward_batch,
                    constants=constants,
                    carry=replace(carry, env_states=env_states_batch),
                    on_policy_variables=on_policy_variables_batch,
                    rng=batch_rng,
                )
            
            # Update the carry's shared states
            carry = replace(
                carry,
                opt_state=next_carry.opt_state,
                shared_state=next_carry.shared_state,
            )
            
            return carry, (metrics, logged_traj)
        
        def update_model_across_batches(
            carry: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
            # Shuffles the batch indices
            batch_indices = jax.random.permutation(rng, self.config.num_envs)
            batch_indices = batch_indices[: self.config.batch_size * (self.config.num_envs // self.config.batch_size)]
            batch_indices = batch_indices.reshape(-1, self.config.batch_size)
            
            # Loops over the batches
            batch_rngs = jax.random.split(rng, len(batch_indices))
            carry, (metrics_list, logged_traj_list) = xax.scan(
                update_model_in_batch,
                carry,
                (batch_indices, batch_rngs),
                jit_level=JitLevel.RL_CORE,
            )
            
            # Gets the last logged trajectory
            logged_traj = logged_traj_list[-1]
            
            # Averages the metrics
            metrics = jax.tree.map(lambda *args: jnp.mean(jnp.array(args)), *metrics_list)
            
            return carry, metrics, logged_traj
        
        # Loops over the number of passes
        pass_rngs = jax.random.split(rng, self.config.num_passes)
        carry, metrics_list, logged_traj_list = xax.scan(
            update_model_across_batches,
            carry,
            pass_rngs,
            jit_level=JitLevel.RL_CORE,
        )
        
        # Gets the last logged trajectory
        logged_traj = logged_traj_list[-1]
        
        # Averages the metrics
        train_metrics = jax.tree.map(lambda *args: jnp.mean(jnp.array(args)), *metrics_list)
        
        return carry, train_metrics, logged_traj
    
    def _distributed_single_step(
        self,
        *,
        trajectories: Trajectory,
        rewards: RewardState,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """Distributed version of single step using pmap for gradient computation."""
        
        # Reshape data for device distribution
        trajectories_distributed = _reshape_for_devices(trajectories, self.num_devices)
        rewards_distributed = _reshape_for_devices(rewards, self.num_devices)
        on_policy_variables_distributed = _reshape_for_devices(on_policy_variables, self.num_devices)
        
        # Replicate carry across devices
        carry_replicated = _device_put_replicated(carry)
        
        # Define pmapped gradient computation
        @jax.pmap(axis_name=_PMAP_AXIS_NAME)
        def compute_gradients_and_metrics(
            traj_shard: Trajectory,
            rewards_shard: RewardState,
            on_policy_vars_shard: PPOVariables,
            carry_shard: RLLoopCarry,
            rng_shard: PRNGKeyArray,
        ) -> tuple[tuple, xax.FrozenDict[str, Array], LoggedTrajectory]:
            """Compute gradients on each device shard."""
            
            # Get PPO metrics and gradients for this shard
            grads, metrics = self._get_ppo_metrics_and_grads(
                trajectories=traj_shard,
                rewards=rewards_shard,
                constants=constants,
                carry=carry_shard,
                on_policy_variables=on_policy_vars_shard,
                rng=rng_shard,
            )
            
            # Average gradients across devices
            grads = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name=_PMAP_AXIS_NAME), grads)
            
            # Create logged trajectory (use first trajectory for logging)
            logged_traj = LoggedTrajectory(
                observations=traj_shard.obs,
                commands=traj_shard.command,
                actions=traj_shard.action,
                rewards=rewards_shard,
            )
            
            return grads, metrics, logged_traj
        
        # Distribute RNG across devices
        rngs_distributed = jax.random.split(rng, self.num_devices)
        
        # Compute gradients in parallel
        grads_list, metrics_list, logged_traj_list = compute_gradients_and_metrics(
            trajectories_distributed,
            rewards_distributed,
            on_policy_variables_distributed,
            carry_replicated,
            rngs_distributed,
        )
        
        # Use gradients from first device (they should be averaged already)
        grads = _unreplicate_first_device(grads_list)
        metrics = _unreplicate_first_device(metrics_list)
        logged_traj = _unreplicate_first_device(logged_traj_list)
        
        # Apply gradients with clipping
        carry = self.apply_gradients_with_clipping(
            gradients=grads,
            constants=constants,
            carry=carry,
        )
        
        return carry, metrics, logged_traj