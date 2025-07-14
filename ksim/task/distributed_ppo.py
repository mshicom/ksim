# ksim/task/distributed_ppo.py
"""Defines a distributed task for training with PPO."""

from typing import Generic, TypeVar
import jax
import jax.numpy as jnp
import xax
import equinox as eqx
from dataclasses import dataclass, replace
from jaxtyping import Array, PRNGKeyArray, PyTree

import ksim
from ksim.task.distributed_rl import DistributedRLTask, DistributedRLConfig, _PMAP_AXIS_NAME
from ksim.task.ppo import PPOTask, PPOConfig, PPOVariables
from ksim.debugging import JitLevel
from ksim.types import LoggedTrajectory, RewardState, Trajectory
from ksim.task.rl import (
    RolloutConstants,
    RolloutSharedState,
    RolloutEnvState,
    LoggedTrajectory,
    RLLoopConstants,
    RLLoopCarry,
)
import logging
logger = logging.getLogger(__name__)

@jax.tree_util.register_dataclass
@dataclass
class DistributedPPOConfig(DistributedRLConfig, PPOConfig):
    dummy_ppo_config_field: int = 0 # Dummy field to ensure this class is not empty


Config = TypeVar("Config", bound=DistributedPPOConfig)


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
        step_count: int,  # Added parameter for gradient sync control
    ) -> tuple[xax.FrozenDict[str, Array], LoggedTrajectory, PyTree]:
        """Override to add gradient synchronization with configurable frequency."""
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
        
        # Conditionally synchronize gradients
        should_sync = step_count % self.config.gradient_sync_period == 0
        
        def sync_fn(g):
            return jax.lax.cond(
                should_sync,
                lambda x: jax.lax.pmean(x, axis_name=_PMAP_AXIS_NAME),
                lambda x: x,
                g
            )
        
        synced_grads = jax.tree.map(sync_fn, grads)
        
        return metrics, logged_trajectory, synced_grads
    
    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.RL_CORE)
    def pmapped_update_model(
        self,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
        step_count: int,  # Added parameter for gradient sync control
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """pmap-compatible version of update_model for distributed PPO training."""
        # Gets the policy model.
        policy_model_arr = carry.shared_state.model_arrs[0]
        policy_model_static = constants.constants.model_statics[0]
        policy_model = eqx.combine(policy_model_arr, policy_model_static)

        # Runs the policy model on the trajectory to get the PPO variables.
        # Note: We're operating on a slice of the environments now
        num_envs_per_device = trajectories.done.shape[0]
        on_policy_rngs = jax.random.split(rng, num_envs_per_device)
        
        ppo_fn = xax.vmap(
            self.get_ppo_variables, 
            in_axes=(None, 0, 0, 0), 
            jit_level=JitLevel.RL_CORE
        )
        
        on_policy_variables, _ = ppo_fn(
            policy_model, 
            trajectories, 
            carry.env_states.model_carry, 
            on_policy_rngs
        )
        
        on_policy_variables = jax.tree.map(
            lambda x: jax.lax.stop_gradient(x), 
            on_policy_variables
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

            # Override _single_step to use the distributed version with step_count
            next_carry, metrics, logged_traj = self._single_step_distributed(
                trajectories=trajectory_batch,
                rewards=reward_batch,
                constants=constants,
                carry=replace(carry, env_states=env_states_batch),
                on_policy_variables=on_policy_variables_batch,
                rng=batch_rng,
                step_count=step_count,
            )

            # Update the carry's shared states.
            carry = replace(
                carry,
                opt_state=next_carry.opt_state,
                shared_state=next_carry.shared_state,
            )

            return carry, (metrics, logged_traj)

        # Define device-local batch size and number of batches
        batch_size = self.config.batch_size // self.get_num_local_devices()
        num_batches = num_envs_per_device // batch_size

        # Applies N steps of gradient updates.
        def update_model_across_batches(
            carry: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            shuffle_rng, batch_rng = jax.random.split(rng)

            # Shuffle the indices so that minibatch updates are different.
            indices = jnp.arange(num_envs_per_device)
            indices = jax.random.permutation(shuffle_rng, indices, independent=False)
            indices_by_batch = indices.reshape(num_batches, batch_size)

            carry, (metrics, trajs_for_logging) = xax.scan(
                update_model_in_batch,
                carry,
                (indices_by_batch, jax.random.split(batch_rng, num_batches)),
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

        # Optionally verify state consistency
        if self.config.verify_state_consistency:
            is_consistent = self.verify_state_consistency(carry.shared_state.model_arrs)
            # Use debug print to report consistency without affecting computation
            jax.debug.print("Model consistency check: {}", is_consistent)

        return carry, metrics, logged_traj
    
    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.RL_CORE)
    def _single_step_distributed(
        self,
        trajectories: Trajectory,
        rewards: RewardState,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
        step_count: int,  # Added parameter for gradient sync control
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """Distributed version of _single_step that passes step_count to gradient computation."""
        # Gets the policy model and optimizer.
        model_arr = carry.shared_state.model_arrs[0]
        model_static = constants.constants.model_statics[0]
        optimizer = constants.optimizer[0]
        opt_state = carry.opt_state[0]

        # Computes the metrics and PPO gradients with step count for sync control
        ppo_metrics, logged_trajectory, grads = self._get_ppo_metrics_and_grads(
            model_arr=model_arr,
            model_static=model_static,
            trajectories=trajectories,
            rewards=rewards,
            init_carry=carry.env_states.model_carry,
            on_policy_variables=on_policy_variables,
            rng=rng,
            step_count=step_count,
        )

        # Applies the gradients with clipping.
        new_model_arr, new_opt_state, grad_metrics = self.apply_gradients_with_clipping(
            model_arr=model_arr,
            grads=grads,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Updates the carry with the new model and optimizer states.
        carry = replace(
            carry,
            shared_state=replace(
                carry.shared_state,
                model_arrs=xax.tuple_insert(carry.shared_state.model_arrs, 0, new_model_arr),
            ),
            opt_state=xax.tuple_insert(carry.opt_state, 0, new_opt_state),
        )

        # Gets the metrics dictionary.
        metrics: xax.FrozenDict[str, Array] = xax.FrozenDict(ppo_metrics.unfreeze() | grad_metrics)

        return carry, metrics, logged_trajectory
    
    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.OUTER_LOOP)
    def _rl_train_loop_step_distributed(
        self,
        carry: RLLoopCarry,
        constants: RLLoopConstants,
        state: xax.State,
        rng: PRNGKeyArray,
        step_count: int,  # Added parameter for gradient sync control
    ) -> tuple[RLLoopCarry, tuple[Trajectory, RewardState], tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]]:
        """Distributed version of _rl_train_loop_step."""
        # Rolls out a new trajectory using pmap instead of vmap
        trajectories, rewards, env_state = self._pmap_single_unroll(
            constants.constants,
            carry.env_states,
            carry.shared_state,
        )
        
        # Update environment states
        carry = replace(
            carry,
            env_states=env_state,
        )
        
        # Run update on the trajectory
        next_carry, train_metrics, logged_traj = self.pmapped_update_model(
            constants=constants,
            carry=carry,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
            step_count=step_count,
        )
        
        # Return everything needed by the outer loop
        return next_carry, (trajectories, rewards), (next_carry, train_metrics, logged_traj)
    
    def _pmap_single_unroll(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RewardState, RolloutEnvState]:
        """pmap-wrapped version of _single_unroll."""
        # Create pmap version of _single_unroll
        pmapped_unroll = jax.pmap(
            self._single_unroll,
            in_axes=(None, 0, None),
            axis_name=_PMAP_AXIS_NAME,
            static_broadcasted_argnums=(0,),
        )
        
        # Call the pmapped function
        return pmapped_unroll(constants, env_states, shared_state)
    
    def run_training(self) -> None:
        """Override run_training to use distributed training with multiple GPUs."""
        import signal
        import traceback
        import sys
        import textwrap
        import time
        import datetime
        from threading import Thread
        from types import FrameType
        
        def on_exit(signum: int, frame: FrameType | None) -> None:
            if self._is_running:
                self._is_running = False
                if xax.is_master():
                    xax.show_info("Gracefully shutting down distributed training...", important=True)

        signal.signal(signal.SIGINT, on_exit)
        signal.signal(signal.SIGTERM, on_exit)

        with self:
            rng = self.prng_key()
            self.set_loggers()
            self._is_running = True

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Log device information
            num_devices = self.get_num_local_devices()
            process_count = jax.process_count()
            process_id = jax.process_index()
            logger.log(
                xax.LOG_PING, 
                f"Training on {num_devices} devices per host, {process_count} processes (id {process_id})"
            )
                
            # Loads the Mujoco model and logs some information about it
            mj_model: ksim.PhysicsModel = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            metadata = self.get_mujoco_model_metadata(mj_model)
            ksim.utils.mujoco.log_joint_config_table(mj_model, metadata, self.logger)

            # Initialize training state in a distributed manner
            constants, carry, state = self.initialize_rl_training(mj_model, rng)
            
            # Create pmapped version of rl_train_loop_step
            pmapped_train_step = jax.pmap(
                lambda c, const, s, r, step_count: self._rl_train_loop_step_distributed(
                    c, const, s, r, step_count
                ),
                in_axes=(0, None, None, 0, None),
                axis_name=_PMAP_AXIS_NAME,
                static_broadcasted_argnums=(1, 2),
            )

            # Check for weak types that could slow down compilation
            for name, leaf in xax.get_named_leaves(carry, max_depth=3):
                aval = jax.core.get_aval(leaf)
                if hasattr(aval, 'weak_type') and aval.weak_type:
                    self.logger.warning(
                        "Found weak type: '%s' This could slow down compilation time", 
                        name
                    )

            # Creates the markers for rendering
            markers = self.get_markers(
                commands=constants.constants.commands,
                observations=constants.constants.observations,
                rewards=constants.constants.rewards,
            )

            # Creates the viewer for rendering
            viewer = ksim.task.rl.get_default_viewer(
                mj_model=mj_model,
                config=self.config,
            )

            state = self.on_training_start(state)

            is_first_step = True
            last_full_render_time = 0.0
            step_count = 0  # Counter for tracking sync periods
            
            # Replicate the random key across devices
            rngs = jax.random.split(rng, num_devices)
            
            try:
                while self._is_running and not self.is_training_over(state):
                    # Runs the training loop
                    with xax.ContextTimer() as timer:
                        valid_step = self.valid_step_timer(state)

                        state = state.replace(
                            phase="valid" if valid_step else "train",
                        )

                        state = self.on_step_start(state)
                        
                        # Split RNG for each device
                        rngs, update_rngs = jax.vmap(jax.random.split)(rngs)
                        
                        # Run distributed training step
                        next_carry, (trajectories, rewards), (_, metrics_tuple, logged_trajs) = pmapped_train_step(
                            carry,
                            constants,
                            state,
                            update_rngs,
                            step_count,
                        )
                        step_count += 1
                        
                        # Update carry
                        carry = next_carry
                        
                        # Gather metrics from all devices (take mean across devices)
                        metrics = jax.tree.map(
                            lambda x: jnp.mean(x, axis=0),
                            metrics_tuple
                        )
                        
                        # Get the logged trajectory from the first device
                        logged_traj = jax.tree.map(
                            lambda x: x[0],
                            logged_trajs
                        )

                        if self.config.profile_memory:
                            carry = jax.block_until_ready(carry)
                            metrics = jax.block_until_ready(metrics)
                            logged_traj = jax.block_until_ready(logged_traj)
                            jax.profiler.save_device_memory_profile(
                                self.exp_dir / "distributed_train_loop_step.prof"
                            )

                        # Log metrics
                        self.log_train_metrics(metrics)
                        self.log_state_timers(state)

                        # Synchronize before checkpointing to ensure consistency
                        if self.should_checkpoint(state):
                            # Synchronize hosts before saving
                            if process_count > 1:
                                x = jnp.ones([num_devices])
                                x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
                                assert x[0] == jax.device_count(), "Host synchronization failed before checkpoint"
                                
                            # Save model
                            self._save(constants=constants, carry=carry, state=state)

                        state = self.on_step_end(state)

                        if valid_step and process_id == 0:
                            cur_time = time.monotonic()
                            full_render = cur_time - last_full_render_time > self.config.render_full_every_n_seconds
                            self._log_logged_trajectory_video(
                                logged_traj=logged_traj,
                                markers=markers,
                                viewer=viewer,
                                key="full trajectory" if full_render else "trajectory",
                            )
                            if full_render:
                                self._log_logged_trajectory_graphs(
                                    logged_traj=logged_traj,
                                    log_callback=lambda key, value, namespace: self.logger.log_image(
                                        key=key,
                                        value=value,
                                        namespace=namespace,
                                    ),
                                )
                                last_full_render_time = cur_time

                        # Updates the step and sample counts
                        num_steps = self.config.epochs_per_log_step
                        num_samples = self.rollout_num_samples * self.config.epochs_per_log_step

                        state = state.replace(
                            num_steps=state.num_steps + num_steps,
                            num_samples=state.num_samples + num_samples,
                        )

                        self.write_logs(state)

                    # Update state with the elapsed time
                    elapsed_time = timer.elapsed_time
                    state = state.replace(
                        elapsed_time_s=state.elapsed_time_s + elapsed_time,
                    )

                    if is_first_step:
                        is_first_step = False
                        self.logger.log(
                            xax.LOG_STATUS,
                            "First distributed step time: %s",
                            xax.format_timedelta(datetime.timedelta(seconds=elapsed_time), short=True),
                        )

                # Save the final checkpoint when done
                if process_id == 0:  # Only save from the master process
                    self._save(constants=constants, carry=carry, state=state)

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = f"Finished distributed training after {state.num_steps} steps and {state.num_samples} samples"
                    xax.show_info(msg, important=True)
                if process_id == 0:
                    self._save(constants=constants, carry=carry, state=state)

            except BaseException:
                exception_tb = textwrap.indent(
                    xax.highlight_exception_message(traceback.format_exc()), "  "
                )
                sys.stdout.write(f"Caught exception during distributed training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                if process_id == 0:
                    self._save(constants=constants, carry=carry, state=state)

            finally:
                # Synchronize all hosts before finishing
                if process_count > 1:
                    try:
                        x = jnp.ones([num_devices])
                        x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
                    except:
                        pass
                    
                state = self.on_training_end(state)