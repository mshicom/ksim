"""Distributed multi-GPU training support for KSIM RL tasks."""

__all__ = [
    "DistributedRLTask",
    "DistributedPPOTask",
]

import logging
from dataclasses import replace
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.debugging import JitLevel
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.task.rl import RLConfig, RLLoopCarry, RLLoopConstants, RLTask
from ksim.types import LoggedTrajectory, Metrics

logger = logging.getLogger(__name__)

# Pmap axis name for multi-device parallelism (following Brax convention)
_PMAP_AXIS_NAME = "i"

ConfigType = TypeVar("ConfigType", bound=PPOConfig)
RLConfigType = TypeVar("RLConfigType", bound=RLConfig)


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


def _device_put_replicated(tree: PyTree) -> PyTree:
    """Put a tree of arrays on devices, replicated across all devices."""
    devices = jax.local_devices()
    if len(devices) == 1:
        return tree
    return jax.device_put_replicated(tree, devices)


def _reshape_for_devices(tree: PyTree, num_devices: int) -> PyTree:
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


class DistributedRLTask(RLTask[RLConfigType], Generic[RLConfigType]):
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
            def pmapped_unroll_fn(constants: PyTree, env_states: PyTree, shared_state: PyTree) -> PyTree:
                return xax.vmap(
                    self._single_unroll,
                    in_axes=(None, 0, None),
                    jit_level=JitLevel.UNROLL,
                )(constants, env_states, shared_state)

            pmapped_unroll = jax.pmap(pmapped_unroll_fn, axis_name=_PMAP_AXIS_NAME)

            # Reshape states for device distribution
            carry_reshaped = jax.tree.map(lambda x: _reshape_for_devices(x, self.num_devices), carry_i)

            trajectories, rewards, env_state = pmapped_unroll(
                constants.constants,
                carry_reshaped.env_states,
                carry_reshaped.shared_state,
            )

            # Flatten trajectories and rewards back to single device dimension
            trajectories = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trajectories)
            rewards = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), rewards)
            env_state = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_state)

            # Update carry with new env states
            carry_i = replace(carry_i, env_states=env_state)

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
            carry_i = replace(carry_i, env_states=replace(carry_i.env_states, curriculum_state=curriculum_state))

            return carry_i, (metrics, logged_traj)

        for _ in range(self.config.epochs_per_log_step):
            rng, step_rng = jax.random.split(rng)
            carry, (metrics, logged_traj) = single_step_fn(carry, step_rng)

        return carry, metrics, logged_traj


class DistributedPPOTask(PPOTask[ConfigType], DistributedRLTask[ConfigType], Generic[ConfigType]):
    """Distributed multi-GPU version of PPOTask."""

    # For now, simply inherit all behavior from PPOTask
    # The distributed computation happens in DistributedRLTask._rl_train_loop_step
    pass
