# ksim/task/distributed_rl.py
"""Defines a distributed task interface for training reinforcement learning agents."""

import functools
import time
import datetime
import signal
import sys
import textwrap
import traceback
from threading import Thread
from types import FrameType
from typing import Any, Collection, Generic, TypeVar
from dataclasses import dataclass, replace
from abc import ABC
import logging
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import xax
import equinox as eqx

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
from ksim.debugging import JitLevel
from ksim.types import RewardState, Trajectory, PhysicsModel, Metrics
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.task.rl import get_default_viewer
from ksim.utils.mujoco import log_joint_config_table


# Define axis name for pmap
_PMAP_AXIS_NAME = 'device'


@dataclass
class DistributedRLConfig(RLConfig):
    """Configuration for distributed reinforcement learning tasks."""
    
    verify_state_consistency: bool = xax.field(
        value=True,
        help="Whether to verify that the state is consistent across devices periodically."
    )


Config = TypeVar("Config", bound=DistributedRLConfig)


class DistributedRLTask(RLTask[Config], Generic[Config], ABC):
    """Base class for distributed reinforcement learning tasks."""
    
    def get_num_local_devices(self) -> int:
        """Get the number of local devices to use for training."""
        return jax.local_device_count()
    
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
    
    def synchronize_gradients(self, grads: PyTree) -> PyTree:
        """Synchronize gradients across devices."""
        return jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
    
    def initialize_rl_training(
        self,
        mj_model: PhysicsModel,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopConstants, RLLoopCarry, xax.State]:
        """Initialize training state for distributed training."""
        # Get base initialization
        constants, carry, state = super().initialize_rl_training(mj_model, rng)
        
        # Get number of devices
        num_devices = self.get_num_local_devices()
        num_envs_per_device = self.config.num_envs // num_devices
        
        if self.config.num_envs % num_devices != 0:
            raise ValueError(
                f"Number of environments ({self.config.num_envs}) must be divisible by "
                f"number of devices ({num_devices})."
            )
        
        # Reshape env_states to have leading device axis 
        def reshape_for_pmap(x: Any) -> Any:
            if not hasattr(x, 'shape') or x.shape == ():
                return jnp.repeat(x[None], num_devices)
            return x.reshape(num_devices, num_envs_per_device, *x.shape[1:])        
        sharded_env_states = jax.tree.map(reshape_for_pmap, carry.env_states)
        
        # Replicate and add leading device axis to shared_state
        def replicate_for_pmap(x: Any) -> Any:
            return jnp.repeat(x[None], num_devices, axis=0)
        replicated_shared_state = jax.tree.map(replicate_for_pmap, carry.shared_state)
        
        # Update carry with sharded env_states and replicated shared_state
        distributed_carry = replace(
            carry,
            env_states=sharded_env_states,
            shared_state=replicated_shared_state,
        )
        
        return constants, distributed_carry, state
    
    def _save(
        self,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        state: xax.State,
    ) -> None:
        """Override _save to handle distributed state."""
        # Take the first device's model for saving
        model_arrs = jax.tree.map(lambda x: x[0], carry.shared_state.model_arrs)
        shared_state = replace(carry.shared_state, model_arrs=model_arrs)
        carry = replace(carry, shared_state=shared_state)
        
        # Call the parent _save method
        super()._save(constants, carry, state)
    
    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.OUTER_LOOP)
    def _distributed_train_loop_step(
        self,
        carry: RLLoopCarry,
        constants: RLLoopConstants,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, Metrics, LoggedTrajectory]:
        """Distributed version of the training loop step."""
        # Runs a single step of the distributed RL training loop
        # This method will be pmapped across devices
        
        # Split RNG for different operations
        rng, rollout_rng, update_rng = jax.random.split(rng, 3)
        
        # Rolls out a new trajectory
        trajectory, rewards, next_env_state = self._single_unroll(
            constants=constants.constants,
            env_states=carry.env_states,
            shared_state=carry.shared_state,
        )
        
        # Update the model on the trajectory
        next_carry, train_metrics, logged_traj = self.update_model(
            constants=constants,
            carry=replace(carry, env_states=next_env_state),
            trajectories=trajectory,
            rewards=rewards,
            rng=update_rng,
        )
        
        # Steps the curriculum
        curriculum_state = constants.constants.curriculum(
            trajectory=trajectory,
            rewards=rewards,
            training_state=state,
            prev_state=next_env_state.curriculum_state,
        )
        
        # Update the curriculum state
        next_carry = replace(
            next_carry,
            env_states=replace(
                next_carry.env_states,
                curriculum_state=curriculum_state,
            ),
        )
        
        # Convert any array with more than one element to a histogram
        metrics = jax.tree.map(self._histogram_fn, train_metrics)
        
        metrics = Metrics(
            train=train_metrics,
            reward=xax.FrozenDict(self.get_reward_metrics(trajectory, rewards)),
            termination=xax.FrozenDict(self.get_termination_metrics(trajectory)),
            curriculum_level=next_carry.env_states.curriculum_state.level,
        )
        
        return next_carry, metrics, logged_traj
    
    def run_training(self) -> None:
        """Override run_training to use distributed training with multiple GPUs."""
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

            # Get device information
            num_devices = self.get_num_local_devices()
            process_count = jax.process_count()
            process_id = jax.process_index()
            logger.log(
                xax.LOG_PING, 
                f"Training on {num_devices} devices per host, {process_count} processes (id {process_id})"
            )
            
            # Ensure num_envs is divisible by number of devices
            if self.config.num_envs % num_devices != 0:
                raise ValueError(
                    f"Number of environments ({self.config.num_envs}) must be divisible by "
                    f"number of devices ({num_devices})."
                )
                
            # Loads the Mujoco model and logs some information about it
            mj_model: PhysicsModel = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            metadata = self.get_mujoco_model_metadata(mj_model)
            log_joint_config_table(mj_model, metadata, self.logger)

            # Initialize training state in a distributed manner
            constants, carry, state = self.initialize_rl_training(mj_model, rng)
            
            # Create pmapped version of distributed_train_loop_step
            pmapped_train_step = jax.pmap(
                lambda c, const, s, r: self._distributed_train_loop_step(c, const, s, r),
                in_axes=(0, None, None, 0),
                axis_name=_PMAP_AXIS_NAME,
                static_broadcasted_argnums=(1, ),
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
            viewer = get_default_viewer(
                mj_model=mj_model,
                config=self.config,
            )

            state = self.on_training_start(state)

            is_first_step = True
            last_full_render_time = 0.0
            
            try:
                while self._is_running and not self.is_training_over(state):
                    # Runs the training loop
                    with xax.ContextTimer() as timer:
                        valid_step = self.valid_step_timer(state)

                        state = state.replace(
                            phase="valid" if valid_step else "train",
                        )

                        state = self.on_step_start(state)
                        
                        # Update RNG each step
                        rng, update_rng = jax.random.split(rng)
                        # add leading axis for pmap to update_rng
                        update_rngs = jax.random.split(update_rng, num_devices)
                        
                        # Run distributed training step
                        carry, metrics_tuple, logged_trajs = pmapped_train_step(
                            carry,
                            constants,
                            state,
                            update_rngs,
                        )
                        
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