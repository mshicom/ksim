# ksim/task/distributed_ppo.py
"""Defines a distributed task for training with PPO."""

from typing import Generic, TypeVar
import jax
import jax.numpy as jnp
import xax
import equinox as eqx
from dataclasses import replace
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.task.rl import RLLoopCarry, RLLoopConstants

from ksim.task.distributed_rl import DistributedRLTask, DistributedRLConfig, _PMAP_AXIS_NAME
from ksim.task.ppo import PPOTask, PPOVariables
from ksim.debugging import JitLevel
from ksim.types import LoggedTrajectory, RewardState, Trajectory, Metrics


Config = TypeVar("Config", bound=DistributedRLConfig)


class DistributedPPOTask(PPOTask[Config], DistributedRLTask[Config], Generic[Config]):
    """Base class for distributed PPO tasks."""
    
    @xax.jit(static_argnames=["self", "model_static"], jit_level=JitLevel.RL_CORE)
    def _get_ppo_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: RewardState,
        init_carry: PyTree,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[xax.FrozenDict[str, Array], LoggedTrajectory, PyTree]:
        """Override to add gradient synchronization."""
        # Get metrics and gradients using the parent method
        metrics, logged_trajectory, grads = super()._get_ppo_metrics_and_grads(
            model_arr,
            model_static,
            trajectories,
            rewards,
            init_carry,
            on_policy_variables,
            rng,
        )
        
        # Synchronize gradients across devices
        synced_grads = self.synchronize_gradients(grads)
        
        return metrics, logged_trajectory, synced_grads
    
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
        """Override update_model to handle per-device environment counts."""
        # Get the number of environments per device
        num_envs_per_device = trajectories.done.shape[0]
        
        # Gets the policy model.
        policy_model_arr = carry.shared_state.model_arrs[0]
        policy_model_static = constants.constants.model_statics[0]
        policy_model = eqx.combine(policy_model_arr, policy_model_static)

        # Runs the policy model on the trajectory to get the PPO variables.
        # Use num_envs_per_device instead of self.config.num_envs
        on_policy_rngs = jax.random.split(rng, num_envs_per_device)
        ppo_fn = xax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0), jit_level=JitLevel.RL_CORE)
        on_policy_variables, _ = ppo_fn(policy_model, trajectories, carry.env_states.model_carry, on_policy_rngs)
        on_policy_variables = jax.tree.map(lambda x: jax.lax.stop_gradient(x), on_policy_variables)

        # Calculate the batch size and number of batches for this device
        batch_size_per_device = self.config.batch_size // self.get_num_local_devices()
        num_batches_per_device = num_envs_per_device // batch_size_per_device
        
        # Ensure batch size is valid
        if batch_size_per_device <= 0 or num_envs_per_device % batch_size_per_device != 0:
            raise ValueError(
                f"Invalid batch size configuration: num_envs_per_device={num_envs_per_device}, "
                f"batch_size_per_device={batch_size_per_device}"
            )

        # Loops over the trajectory batches and applies gradient updates.
        def update_model_in_batch(
            carry: RLLoopCarry,
            xs: tuple[Array, PRNGKeyArray],
        ) -> tuple[RLLoopCarry, tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            batch_indices, rng = xs
            rng, batch_rng = jax.random.split(rng)

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[batch_indices], trajectories)
            reward_batch = jax.tree.map(lambda x: x[batch_indices], rewards)
            env_states_batch = jax.tree.map(lambda x: x[batch_indices], carry.env_states)
            on_policy_variables_batch = jax.tree.map(lambda x: x[batch_indices], on_policy_variables)

            next_carry, metrics, logged_traj = self._single_step(
                trajectories=trajectory_batch,
                rewards=reward_batch,
                constants=constants,
                carry=replace(carry, env_states=env_states_batch),
                on_policy_variables=on_policy_variables_batch,
                rng=batch_rng,
            )

            # Update the carry's shared states.
            carry = replace(
                carry,
                opt_state=next_carry.opt_state,
                shared_state=next_carry.shared_state,
            )

            return carry, (metrics, logged_traj)

        # Applies N steps of gradient updates.
        def update_model_across_batches(
            carry: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            shuffle_rng, batch_rng = jax.random.split(rng)

            # Shuffle the indices so that minibatch updates are different.
            indices = jnp.arange(num_envs_per_device)  # Use num_envs_per_device
            indices = jax.random.permutation(shuffle_rng, indices, independent=False)
            indices_by_batch = indices.reshape(num_batches_per_device, batch_size_per_device)

            carry, (metrics, trajs_for_logging) = xax.scan(
                update_model_in_batch,
                carry,
                (indices_by_batch, jax.random.split(batch_rng, num_batches_per_device)),
                jit_level=JitLevel.RL_CORE,
            )

            # Each batch saves one trajectory for logging, get the last.
            traj_for_logging = jax.tree.map(lambda x: x[-1], trajs_for_logging)

            return carry, (metrics, traj_for_logging)

        # Applies gradient update across all batches num_passes times.
        carry, (metrics, trajs_for_logging) = xax.scan(
            update_model_across_batches,
            carry,
            xs=jax.random.split(rng, self.config.num_passes),
            jit_level=JitLevel.RL_CORE,
        )

        # Get the last logged trajectory across all full dataset passes.
        logged_traj = jax.tree.map(lambda x: x[-1], trajs_for_logging)

        if carry.env_states.model_carry is not None:
            # Gets the policy model, using the latest model parameters.
            policy_model_arr = carry.shared_state.model_arrs[0]
            policy_model_static = constants.constants.model_statics[0]
            policy_model = eqx.combine(policy_model_arr, policy_model_static)

            # For the next rollout, update the model carry
            # Use num_envs_per_device instead of self.config.num_envs
            off_policy_rngs = jax.random.split(rng, num_envs_per_device)
            _, next_model_carrys = ppo_fn(
                policy_model,
                trajectories,
                carry.env_states.model_carry,
                off_policy_rngs,
            )

            carry = replace(
                carry,
                env_states=replace(
                    carry.env_states,
                    model_carry=next_model_carrys,
                ),
            )
            
        # Apply any additional synchronization between devices if needed
        if self.config.verify_state_consistency:
            is_consistent = self.verify_state_consistency(carry.shared_state.model_arrs)
            # Debug print doesn't affect computation
            jax.debug.print("Model consistency check: {}", is_consistent)

        return carry, metrics, logged_traj