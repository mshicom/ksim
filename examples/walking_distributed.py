"""Distributed humanoid walking task implementation using multi-GPU training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import distrax
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

# Import components from the regular walking example
from examples.walking import (
    Actor,
    Critic,
    HumanoidWalkingTaskConfig,
    Model,
)


@xax.jit(static_argnames=["num_joints", "std"])
def _create_initial_positions_distributed(
    num_joints: int,
    std: float,
    zeros_data: dict[str, float],
    rng: PRNGKeyArray,
    num_envs: int,
) -> Array:
    """Creates initial joint positions for distributed training."""
    key_initial, key_noise = jax.random.split(rng)

    # Create base positions using zeros
    initial_positions = jnp.zeros((num_envs, num_joints))
    for _joint_name, _position in zeros_data.items():
        # This is a simplified version - in reality you'd need joint indexing
        pass  # Joint assignment logic would go here

    # Add noise
    noise = jax.random.normal(key_noise, (num_envs, num_joints)) * std
    return initial_positions + noise


@jax.tree_util.register_dataclass
@dataclass
class HumanoidWalkingDistributedConfig(ksim.DistributedPPOConfig, HumanoidWalkingTaskConfig):
    """Configuration for distributed humanoid walking task."""

    # Override defaults for distributed training
    num_envs: int = xax.field(
        value=16,
        help="The number of training environments to run in parallel (will be sharded across devices).",
    )
    batch_size: int = xax.field(
        value=8,
        help="The number of trajectories to process in each minibatch during gradient updates.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingDistributedConfig)


class HumanoidWalkingDistributedTask(ksim.DistributedPPOTask[Config], Generic[Config]):
    """Distributed humanoid walking task using multi-GPU training."""

    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        return mujoco.MjModel.from_xml_path(mjcf_path)

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        return ksim.Metadata.from_model(
            mj_model,
            kp=100.0,
            kd=5.0,
            armature=1e-4,
            friction=1e-6,
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "torso"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.LinearPushEvent(
                linvel=1.0,
                interval_range=(2.0, 5.0),
            ),
            ksim.JumpEvent(
                jump_height_range=(1.0, 2.0),
                interval_range=(2.0, 5.0),
            ),
            ksim.JointPerturbationEvent(
                std=50.0,
                mask_prct=0.9,
                interval_range=(0.1, 0.15),
                curriculum_range=(1.0, 1.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, zeros={"abdomen_z": 0.0}),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.ProjectedGravityObservation(),
            ksim.TimestepObservation(),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.TrackingLinearVelocityReward(weight=3.0),
            ksim.TrackingAngularVelocityReward(weight=1.0),
            ksim.OrientationReward(weight=1.0),
            ksim.EnergyReward(weight=-0.05),
            ksim.AliveReward(weight=0.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.TorsoHeightTermination(min_height=0.3),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.JoystickCommand(),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.ManualCurriculum(num_levels=1)

    def get_model(self, key: PRNGKeyArray) -> Model:
        """Creates the actor-critic model."""
        return Model(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        """Returns the initial carry for the model."""
        return None

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Distribution:
        # Same as original walking task
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        joystick_cmd_ohe_7 = commands["joystick_command"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_7,  # 7
            ],
            axis=-1,
        )
        return model.forward(obs_n)

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        # Same as original walking task
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        joystick_cmd_ohe_7 = commands["joystick_command"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_7,  # 7
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:
        # Same as original walking task
        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_tj = self.run_actor(model.actor, transition.obs, transition.command)
            log_probs_tj = action_dist_tj.log_prob(transition.action)
            assert isinstance(log_probs_tj, Array)
            return log_probs_tj

        log_probs_tj = jax.vmap(get_log_prob)(trajectory)
        assert isinstance(log_probs_tj, Array)

        values_tj = jax.vmap(self.run_critic, in_axes=(None, 0, 0))(model.critic, trajectory.obs, trajectory.command)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tj,
            values=values_tj.squeeze(-1),
        )

        return ppo_variables, None

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
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=None)

    # Implementation of abstract methods from DistributedPPOTask
    def _get_initial_model(self, rng: PRNGKeyArray) -> Model:
        """Gets the initial model."""
        return self.get_model(rng)

    def _get_initial_env_state(self, rng: PRNGKeyArray) -> ksim.RolloutEnvState:
        """Gets the initial environment state."""
        # This would typically be initialized through the base RL task setup
        # For now, return a placeholder - this would need to be implemented based on
        # the full rollout initialization logic
        return ksim.RolloutEnvState(
            commands=xax.FrozenDict({}),
            physics_state=None,  # Would be properly initialized
            obs_carry=xax.FrozenDict({}),
            reward_carry=xax.FrozenDict({}),
            curriculum_state=None,  # Would be properly initialized
        )

    def _get_initial_model_carry(self) -> None:
        """Gets the initial model carry state."""
        return None  # This model doesn't use carry state

    def _rollout(
        self,
        model: Model,
        env_state: ksim.RolloutEnvState,
        model_carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.Trajectory, ksim.RolloutEnvState, None]:
        """Performs a rollout."""
        # This would implement the actual rollout logic
        # For now, this is a placeholder that would need to be implemented
        # based on the full RL task rollout methodology
        raise NotImplementedError("Full rollout implementation needed")


if __name__ == "__main__":
    # To run distributed training, use:
    #   python -m examples.walking_distributed num_envs=16 batch_size=8
    # To test with simulated devices:
    #   XLA_FLAGS='--xla_force_host_platform_device_count=2' python -m examples.walking_distributed \
    #   num_envs=8 batch_size=4

    config = HumanoidWalkingDistributedConfig(
        # Distributed training parameters
        num_envs=16,  # Will be sharded across available devices
        batch_size=8,
        pmap_axis_name="device",
        grad_sync_period=1,

        # Standard training parameters
        num_passes=2,
        epochs_per_log_step=1,
        rollout_length_seconds=8.0,
        global_grad_clip=2.0,

        # Simulation parameters
        dt=0.002,
        ctrl_dt=0.02,
        iterations=3,
        ls_iterations=5,
        action_latency_range=(0.005, 0.01),
        drop_action_prob=0.01,

        # Checkpointing parameters
        save_every_n_seconds=60,
    )

    task = HumanoidWalkingDistributedTask(config)
    task.run_distributed_training()
