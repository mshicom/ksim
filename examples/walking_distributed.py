"""Distributed humanoid walking task using multi-GPU training."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

import ksim
from ksim.task.distributed_ppo import DistributedPPOConfig, DistributedPPOTask

# Import components from the standard walking example
from examples.walking import (
    Actor, Critic, Model, NUM_JOINTS, NUM_INPUTS, ZEROS
)


@dataclass
class HumanoidWalkingDistributedConfig(DistributedPPOConfig):
    """Configuration for distributed humanoid walking task."""
    
    # Model parameters (inherited from walking example)
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    
    # Reward parameters
    target_linear_velocity: float = xax.field(
        value=2.0,
        help="The linear velocity for the joystick command.",
    )
    target_angular_velocity: float = xax.field(
        value=math.radians(90.0),
        help="The angular velocity for the joystick command.",
    )
    
    # Optimizer parameters
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )
    
    # Distributed training parameters
    pmap_axis_name: str = xax.field(
        value="device",
        help="Axis name for pmapping operations.",
    )
    grad_sync_period: int = xax.field(
        value=1,
        help="Synchronize gradients across devices every N updates.",
    )


class HumanoidWalkingDistributedTask(DistributedPPOTask):
    """Distributed humanoid walking task.
    
    This task extends the basic humanoid walking task with multi-GPU capabilities.
    It shards environments across available devices and synchronizes gradients.
    """
    
    config: HumanoidWalkingDistributedConfig
    
    def get_optimizer(self):
        """Gets the optimizer for model updates."""
        import optax
        return optax.adam(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.adam_weight_decay,
        )
    
    def update_model(self, *args, **kwargs):
        """Placeholder update_model method."""
        # This would be implemented with the actual PPO update logic
        # For now, just return the input model unchanged
        model = args[0] if args else kwargs.get('model')
        return model, {}
    
    def get_mujoco_model(self) -> mujoco.MjModel:
        """Returns the MuJoCo model for the humanoid."""
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        return mujoco.MjModel.from_xml_path(mjcf_path)
    
    def get_mujoco_model_metadata(self) -> ksim.Metadata | None:
        """Returns metadata for the MuJoCo model."""
        return None
        
    def get_model(self, key: PRNGKeyArray) -> Model:
        """Creates the actor-critic model."""
        return Model(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )
    
    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        """Returns position actuators for the humanoid."""
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )
    
    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        """Returns physics randomizers for domain randomization."""
        return [
            ksim.StaticFrictionRandomizer(scale_lower=0.5, scale_upper=2.0),
            ksim.ArmatureRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.MassMultiplicationRandomizer.from_body_name(
                physics_model, "torso", scale_lower=0.95, scale_upper=1.05
            ),
            ksim.JointDampingRandomizer(scale_lower=0.9, scale_upper=1.1),
            ksim.JointZeroPositionRandomizer(scale_lower=-0.01, scale_upper=0.01),
        ]
    
    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        """Returns events for environmental perturbations."""
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.0,
                x_angular_force=0.1,
                y_angular_force=0.1,
                z_angular_force=0.3,
                interval_range=(0.25, 0.75),
            ),
        ]
    
    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        """Returns reset systems for initializing episodes."""
        return [
            ksim.get_xy_position_reset(physics_model),
            ksim.RandomJointPositionReset(scale=0.1, zeros=dict(ZEROS)),
            ksim.RandomBaseVelocityXYReset(scale=0.5),
            ksim.RandomHeadingReset(),
        ]
    
    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        """Returns observations for the agent."""
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(), 
            ksim.ProjectedGravityObservation.create(physics_model, "imu", lag_range=(0.0, 0.05)),
            ksim.ActuatorForceObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
        ]
    
    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        """Returns commands for controlling the agent."""
        return [ksim.JoystickCommand()]
    
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        """Returns reward functions for the task."""
        return [
            ksim.StayAliveReward(scale=1.0),
            ksim.JoystickReward(
                forward_speed=self.config.target_linear_velocity,
                backward_speed=self.config.target_linear_velocity / 2.0,
                strafe_speed=self.config.target_linear_velocity / 2.0,
                rotation_speed=self.config.target_angular_velocity,
                scale=1.0,
            ),
        ]
    
    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        """Returns termination conditions."""
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]
    
    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        """Returns curriculum for progressive learning."""
        return ksim.EpisodeLengthCurriculum(
            num_levels=10,
            increase_threshold=3.0,
            decrease_threshold=1.0,
            min_level_steps=50,
        )
    
    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        """Returns initial model carry state (none for feedforward model)."""
        return None
    
    def sample_action(
        self,
        model: Model,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        """Samples actions from the policy."""
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=None)
    
    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ):
        """Runs the actor to get action distribution.""" 
        # Process observations (simplified version)
        obs_list = []
        
        # Add timestep observation with sin/cos encoding
        timestep = observations["timestep_observation"]
        obs_list.extend([jnp.sin(timestep), jnp.cos(timestep)])
        
        # Add joint positions and velocities
        obs_list.append(observations["joint_position_observation"])
        obs_list.append(observations["joint_velocity_observation"] / 10.0)
        
        # Add other observations
        obs_list.append(observations["center_of_mass_inertia_observation"])
        obs_list.append(observations["center_of_mass_velocity_observation"])
        obs_list.append(observations["projected_gravity_observation"])
        obs_list.append(observations["actuator_force_observation"] / 100.0)
        obs_list.append(observations["base_position_observation"])
        obs_list.append(observations["base_orientation_observation"])
        obs_list.append(observations["base_linear_velocity_observation"])
        obs_list.append(observations["base_angular_velocity_observation"])
        
        # One-hot encode joystick command
        joystick_cmd = commands["joystick_command"]
        one_hot_cmd = jnp.eye(7)[joystick_cmd.astype(int)]
        obs_list.append(one_hot_cmd)
        
        # Concatenate all observations
        obs_cat = jnp.concatenate([jnp.atleast_1d(obs).flatten() for obs in obs_list])
        
        return model(obs_cat)
    
    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        """Runs the critic to get value estimate."""
        # Use same observation processing as actor
        obs_list = []
        
        timestep = observations["timestep_observation"]
        obs_list.extend([jnp.sin(timestep), jnp.cos(timestep)])
        
        obs_list.append(observations["joint_position_observation"])
        obs_list.append(observations["joint_velocity_observation"] / 10.0)
        obs_list.append(observations["center_of_mass_inertia_observation"])
        obs_list.append(observations["center_of_mass_velocity_observation"])
        obs_list.append(observations["projected_gravity_observation"])
        obs_list.append(observations["actuator_force_observation"] / 100.0)
        obs_list.append(observations["base_position_observation"])
        obs_list.append(observations["base_orientation_observation"])
        obs_list.append(observations["base_linear_velocity_observation"])
        obs_list.append(observations["base_angular_velocity_observation"])
        
        joystick_cmd = commands["joystick_command"]
        one_hot_cmd = jnp.eye(7)[joystick_cmd.astype(int)]
        obs_list.append(one_hot_cmd)
        
        obs_cat = jnp.concatenate([jnp.atleast_1d(obs).flatten() for obs in obs_list])
        
        return model(obs_cat)


if __name__ == "__main__":
    config = HumanoidWalkingDistributedConfig(
        # Training parameters
        num_envs=8,  # Will be distributed across available devices
        batch_size=4,
        num_passes=2,
        epochs_per_log_step=1,
        rollout_length_seconds=8.0,
        global_grad_clip=2.0,
        
        # Distributed parameters
        pmap_axis_name="device",
        grad_sync_period=1,
        
        # Simulation parameters
        dt=0.002,
        ctrl_dt=0.02,
        iterations=3,
        ls_iterations=5,
        action_latency_range=(0.005, 0.01),
        drop_action_prob=0.01,
        
        # Model parameters
        hidden_size=128,
        depth=5,
        num_mixtures=5,
        learning_rate=1e-3,
    )
    
    task = HumanoidWalkingDistributedTask(config)
    
    # Verify distributed setup
    task.verify_distributed_setup()
    
    # For testing, we'll just verify the setup rather than run full training
    print("✓ Distributed walking task created successfully")
    print(f"✓ Using {task.get_device_count()} devices")
    print(f"✓ {task.get_envs_per_device()} environments per device")
    
    # Note: To run actual distributed training, call:
    # task.run_distributed_training()