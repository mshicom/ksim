# examples/walking_distributed.py
"""Example of distributed training for the walking humanoid task."""

import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

from typing import Generic, TypeVar
from dataclasses import dataclass
from ksim.task.ppo import PPOConfig
from ksim.task.distributed_ppo import DistributedPPOTask, DistributedPPOConfig
from examples.walking import HumanoidWalkingTaskConfig, HumanoidWalkingTask

@dataclass
class DistributedHumanoidWalkingConfig(HumanoidWalkingTaskConfig, DistributedPPOConfig):
    """Config for the distributed humanoid walking task."""
    
    # Distribution-specific parameters
    max_devices_per_host: int | None = None  # Use all available devices
    gradient_sync_period: int = 1  # Sync gradients every step
    verify_state_consistency: bool = True


Config = TypeVar("Config", bound=DistributedHumanoidWalkingConfig)


class DistributedHumanoidWalkingTask(DistributedPPOTask[Config], Generic[Config]):
    """Distributed version of the humanoid walking task."""
    
    # Inherit most methods from HumanoidWalkingTask
    get_optimizer = HumanoidWalkingTask.get_optimizer
    get_mujoco_model = HumanoidWalkingTask.get_mujoco_model
    get_mujoco_model_metadata = HumanoidWalkingTask.get_mujoco_model_metadata
    get_actuators = HumanoidWalkingTask.get_actuators
    get_physics_randomizers = HumanoidWalkingTask.get_physics_randomizers
    get_events = HumanoidWalkingTask.get_events
    get_resets = HumanoidWalkingTask.get_resets
    get_observations = HumanoidWalkingTask.get_observations
    get_commands = HumanoidWalkingTask.get_commands
    get_rewards = HumanoidWalkingTask.get_rewards
    get_terminations = HumanoidWalkingTask.get_terminations
    get_curriculum = HumanoidWalkingTask.get_curriculum
    get_model = HumanoidWalkingTask.get_model
    get_initial_model_carry = HumanoidWalkingTask.get_initial_model_carry
    run_actor = HumanoidWalkingTask.run_actor
    run_critic = HumanoidWalkingTask.run_critic
    get_ppo_variables = HumanoidWalkingTask.get_ppo_variables
    sample_action = HumanoidWalkingTask.sample_action

    # Only override methods that need to be distributed


if __name__ == "__main__":
    # Print device information
    print(f"JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"Process count: {jax.process_count()}")
    
    # Launch the distributed training
    DistributedHumanoidWalkingTask.launch(
        DistributedHumanoidWalkingConfig(
            max_steps=3,
            # Training parameters
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=0.1,
            global_grad_clip=2.0,
            # Logging parameters
            valid_first_n_steps=1,
            # Simulation parameters
            dt=0.002,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            action_latency_range=(0.005, 0.01),
            drop_action_prob=0.01,
            # Checkpointing parameters
            save_every_n_seconds=60,
            # Distribution parameters
            max_devices_per_host=None,  # Use all available devices
            gradient_sync_period=1,     # Sync gradients every step
        ),
    )