"""Tests for RLTask's step_engine method with vmap and pmap operations."""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from typing import Collection, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
import attrs
from dataclasses import dataclass

from ksim.actuators import Actuators, TorqueActuators
from ksim.commands import Command, FloatVectorCommand
from ksim.curriculum import Curriculum, ConstantCurriculum
from ksim.events import Event
from ksim.observation import Observation, BasePositionObservation
from ksim.randomization import PhysicsRandomizer, JointDampingRandomizer
from ksim.terminations import EpisodeLengthTermination
from ksim.resets import Reset
from ksim.rewards import Reward, StayAliveReward
from ksim.task.rl import (
    RLConfig,
    RLTask,
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
    get_initial_commands,
    get_initial_obs_carry,
    get_initial_reward_carry,
    get_physics_randomizers,
)
from ksim.terminations import Termination
from ksim.types import Action, PhysicsModel, PhysicsState, PhysicsData, Trajectory
from ksim.task.rl import RolloutConstants, RolloutEnvState, RolloutSharedState



@pytest.fixture
def rng() -> jax.Array:
    """Return a random number generator key."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def simple_model() -> mujoco.MjModel:
    """Create a simple model with 3 bodies and 3 joints for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" />
            <body name="torso" pos="0 0 0.5">
                <joint name="free_joint" type="free"/>
                <geom name="torso_geom" type="sphere" size="0.1" />
                <body name="body1" pos="0.2 0 0">
                    <joint name="joint1" type="hinge" axis="0 0 1" damping="0.1" armature="0.1" frictionloss="0.1" />
                    <geom name="body1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    <body name="body2" pos="0.2 0 0">
                        <joint name="joint2" type="hinge" axis="0 1 0"
                        damping="0.1" armature="0.1" frictionloss="0.1" />
                        <geom name="body2_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)
    # Set reasonable simulation parameters
    mj_model.opt.timestep = 0.002  # dt
    return mj_model


class DummyModel(eqx.Module):
    """Dummy model for testing purposes."""

    def __call__(self, obs: Array) -> Array:
        """Return a zero action."""
        return jnp.zeros((1,))


@dataclass
class TestRLConfig(RLConfig):
    """Minimal RL config for testing."""

    num_envs: int = xax.field(value=8, help="Number of environments")
    batch_size: int = xax.field(value=4, help="Batch size")
    rollout_length_seconds: float = xax.field(value=0.1, help="Rollout length in seconds")
    ctrl_dt: float = xax.field(value=0.02, help="Control timestep")
    dt: float = xax.field(value=0.002, help="Physics timestep")
    iterations: int = xax.field(value=1, help="Number of iterations")
    ls_iterations: int = xax.field(value=1, help="Line search iterations")


@attrs.define(frozen=True, kw_only=True)
class TestReset(Reset):
    """Simple test reset that doesn't modify the data."""

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        """Return data unchanged."""
        return data

class TestRLTask(RLTask[TestRLConfig]):
    """Minimal RL task for testing vmap and pmap operations."""

    def __init__(self, config: TestRLConfig) -> None:
        super().__init__(config)
        self.dummy_model = DummyModel()

    def get_mujoco_model(self) -> mujoco.MjModel:
        """Return a simple test model."""
        xml = """
        <mujoco>
            <worldbody>
                <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" />
                <body name="torso" pos="0 0 0.5">
                    <joint name="free_joint" type="free"/>
                    <geom name="torso_geom" type="sphere" size="0.1" />
                    <body name="body1" pos="0.2 0 0">
                        <joint name="joint1" type="hinge" axis="0 0 1" />
                        <geom name="body1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        mj_model = mujoco.MjModel.from_xml_string(xml)
        return self.set_mujoco_model_opts(mj_model)

    def get_physics_randomizers(self, physics_model: PhysicsModel) -> Collection[PhysicsRandomizer]:
        """Return randomizers for testing."""
        return [JointDampingRandomizer(scale_lower=0.9, scale_upper=1.1)]

    def get_resets(self, physics_model: PhysicsModel) -> Collection[Reset]:
        """Return resets for testing."""
        return [TestReset()]

    def get_events(self, physics_model: PhysicsModel) -> Collection[Event]:
        """Return events for testing."""
        return []

    def get_actuators(self, physics_model: PhysicsModel, metadata=None) -> Actuators:
        """Return actuators for testing."""
        return TorqueActuators()

    def get_observations(self, physics_model: PhysicsModel) -> Collection[Observation]:
        """Return observations for testing."""
        return [BasePositionObservation()]

    def get_commands(self, physics_model: PhysicsModel) -> Collection[Command]:
        """Return commands for testing."""
        return [FloatVectorCommand(ranges=((0.0, 0.0), (0.0, 0.0)))]

    def get_rewards(self, physics_model: PhysicsModel) -> Collection[Reward]:
        """Return rewards for testing."""
        return [StayAliveReward(scale=1.0)]

    def get_terminations(self, physics_model: PhysicsModel) -> Collection[Termination]:
        """Return terminations for testing."""
        return [EpisodeLengthTermination(max_length_sec=1.0)]

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> PyTree | None:
        """Return initial model carry."""
        return None

    def get_curriculum(self, physics_model: PhysicsModel) -> Curriculum:
        """Return curriculum for testing."""
        return ConstantCurriculum(level=1.0)

    def sample_action(
        self,
        model: PyTree,
        model_carry: PyTree,
        physics_model: PhysicsModel,
        physics_state: PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> Action:
        """Return a simple action for testing."""
        # Always return a constant action for predictable testing
        return Action(
            action=jnp.ones((1,)),  # 1 actuator for the hinge joint
            carry=model_carry,
            aux_outputs={},
        )

    def update_model(self, **kwargs) -> tuple[Any, xax.FrozenDict[str, Array], Any]:  # type: ignore
        """Dummy update method."""
        raise NotImplementedError("Not needed for testing step_engine")
    def get_model(self, obs_size: int, action_size: int, rng: PRNGKeyArray) -> Any:
        """Return the dummy model."""
        return self.dummy_model
    def get_optimizer(self) -> Any:
        """Return a dummy optimizer."""
        return None


def test_step_engine_basic(simple_model, rng):
    """Test that step_engine works correctly for a single environment."""
    config = TestRLConfig()
    task = TestRLTask(config)
    
    # Create MJX model
    mjx_model = task.get_mjx_model(simple_model)
    
    # Create constants
    dummy_model = DummyModel()
    model_arrs, model_statics = eqx.partition(dummy_model, task.model_partition_fn)
    
    constants = RolloutConstants(
        model_statics=(model_statics,),
        engine=task.get_engine(mjx_model),
        observations=tuple(task.get_observations(mjx_model)),
        commands=tuple(task.get_commands(mjx_model)),
        rewards=tuple(task.get_rewards(mjx_model)),
        terminations=tuple(task.get_terminations(mjx_model)),
        curriculum=task.get_curriculum(mjx_model),
        argmax_action=True,
        aux_constants=xax.FrozenDict({}),
    )
    
    # Create env state
    rng, init_rng = jax.random.split(rng)
    
    # Setup single environment state
    curriculum_state = constants.curriculum.get_initial_state(init_rng)
    physics_state = constants.engine.reset(mjx_model, curriculum_state.level, init_rng)
    
    randomizations = get_physics_randomizers(mjx_model, task.get_physics_randomizers(mjx_model), init_rng)
    commands = get_initial_commands(init_rng, physics_state.data, constants.commands, curriculum_state.level)
    obs_carry = get_initial_obs_carry(init_rng, physics_state, constants.observations)
    reward_carry = get_initial_reward_carry(init_rng, constants.rewards)
    model_carry = task.get_initial_model_carry(init_rng)
    
    env_state = RolloutEnvState(
        commands=commands,
        physics_state=physics_state,
        randomization_dict=randomizations,
        model_carry=model_carry,
        reward_carry=reward_carry,
        obs_carry=obs_carry,
        curriculum_state=curriculum_state,
        rng=init_rng,
    )
    
    # Create shared state
    shared_state = RolloutSharedState(
        physics_model=mjx_model,
        model_arrs=(model_arrs,),
        aux_values=xax.FrozenDict({}),
    )
    
    # Test step_engine
    trajectory, next_env_state = task.step_engine(
        constants=constants,
        env_states=env_state,
        shared_state=shared_state,
    )
    
    # Verify trajectory and state
    assert isinstance(trajectory, Trajectory)
    assert isinstance(next_env_state, RolloutEnvState)
    
    # Check that action was applied correctly
    assert jnp.allclose(trajectory.action, jnp.ones((1,)))


def test_vmap_step_engine(simple_model, rng):
    """Test that vmap works correctly on step_engine."""
    config = TestRLConfig(num_envs=4)
    task = TestRLTask(config)
    
    # Create MJX model
    mjx_model = task.get_mjx_model(simple_model)
    
    # Create constants
    dummy_model = DummyModel()
    model_arrs, model_statics = eqx.partition(dummy_model, task.model_partition_fn)
    
    constants = RolloutConstants(
        model_statics=(model_statics,),
        engine=task.get_engine(mjx_model),
        observations=tuple(task.get_observations(mjx_model)),
        commands=tuple(task.get_commands(mjx_model)),
        rewards=tuple(task.get_rewards(mjx_model)),
        terminations=tuple(task.get_terminations(mjx_model)),
        curriculum=task.get_curriculum(mjx_model),
        argmax_action=True,
        aux_constants=xax.FrozenDict({}),
    )
    
    # Create env states - batched across 4 environments
    rngs = jax.random.split(rng, config.num_envs)
    
    def create_env_state(init_rng):
        # Setup environment state
        curriculum_state = constants.curriculum.get_initial_state(init_rng)
        physics_state = constants.engine.reset(mjx_model, curriculum_state.level, init_rng)
        
        randomizations = get_physics_randomizers(mjx_model, task.get_physics_randomizers(mjx_model), init_rng)
        commands = get_initial_commands(init_rng, physics_state.data, constants.commands, curriculum_state.level)
        obs_carry = get_initial_obs_carry(init_rng, physics_state, constants.observations)
        reward_carry = get_initial_reward_carry(init_rng, constants.rewards)
        model_carry = task.get_initial_model_carry(init_rng)
        
        return RolloutEnvState(
            commands=commands,
            physics_state=physics_state,
            randomization_dict=randomizations,
            model_carry=model_carry,
            reward_carry=reward_carry,
            obs_carry=obs_carry,
            curriculum_state=curriculum_state,
            rng=init_rng,
        )
    
    # Create vectorized environment states
    vmap_create_env_state = jax.vmap(create_env_state)
    env_states = vmap_create_env_state(rngs)
    
    # Create shared state
    shared_state = RolloutSharedState(
        physics_model=mjx_model,
        model_arrs=(model_arrs,),
        aux_values=xax.FrozenDict({}),
    )
    
    # Create vmapped step_engine
    def step_wrapper(constants, env_states, shared_state):
        return task.step_engine(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
        )

    vmapped_step_engine = jax.vmap(step_wrapper, in_axes=(None, 0, None))

    # Test vmapped step_engine
    trajectories, next_env_states = vmapped_step_engine(
        constants, env_states, shared_state
    )
    
    # Verify trajectories and states
    assert trajectories.action.shape[0] == config.num_envs
    assert isinstance(next_env_states, RolloutEnvState)
    
    # Check that each environment got unique but expected actions
    for i in range(config.num_envs):
        assert jnp.allclose(trajectories.action[i], jnp.ones((1,)))


def test_pmap_step_engine(simple_model, rng):
    """Test that pmap works correctly on step_engine."""
    config = TestRLConfig(num_envs=16)  # One env per device
    task = TestRLTask(config)
    
    # Create MJX model
    mjx_model = task.get_mjx_model(simple_model)
    
    # Create constants
    dummy_model = DummyModel()
    model_arrs, model_statics = eqx.partition(dummy_model, task.model_partition_fn)
    
    constants = RolloutConstants(
        model_statics=(model_statics,),
        engine=task.get_engine(mjx_model),
        observations=tuple(task.get_observations(mjx_model)),
        commands=tuple(task.get_commands(mjx_model)),
        rewards=tuple(task.get_rewards(mjx_model)),
        terminations=tuple(task.get_terminations(mjx_model)),
        curriculum=task.get_curriculum(mjx_model),
        argmax_action=True,
        aux_constants=xax.FrozenDict({}),
    )
    
    # Create env states - one per device
    rngs = jax.random.split(rng, jax.device_count())
    
    def create_env_state(init_rng):
        # Setup environment state
        curriculum_state = constants.curriculum.get_initial_state(init_rng)
        physics_state = constants.engine.reset(mjx_model, curriculum_state.level, init_rng)
        
        randomizations = get_physics_randomizers(mjx_model, task.get_physics_randomizers(mjx_model), init_rng)
        commands = get_initial_commands(init_rng, physics_state.data, constants.commands, curriculum_state.level)
        obs_carry = get_initial_obs_carry(init_rng, physics_state, constants.observations)
        reward_carry = get_initial_reward_carry(init_rng, constants.rewards)
        model_carry = task.get_initial_model_carry(init_rng)
        
        return RolloutEnvState(
            commands=commands,
            physics_state=physics_state,
            randomization_dict=randomizations,
            model_carry=model_carry,
            reward_carry=reward_carry,
            obs_carry=obs_carry,
            curriculum_state=curriculum_state,
            rng=init_rng,
        )
    
    # Create environment states for each device
    env_states = jax.pmap(lambda rng: create_env_state(rng))(rngs)
    
    # Create shared state (replicated across devices)
    shared_state = RolloutSharedState(
        physics_model=mjx_model,
        model_arrs=(model_arrs,),
        aux_values=xax.FrozenDict({}),
    )
    
    # Create pmapped step_engine
    def step_wrapper(env_state, shared_state):
        return task.step_engine(
            constants=constants,
            env_states=env_state,
            shared_state=shared_state,
        )
    
    pmapped_step_engine = jax.pmap(step_wrapper)
    
    # Test pmapped step_engine
    trajectories, next_env_states = pmapped_step_engine(
        env_states,
        jax.tree.map(lambda x: jnp.array([x] * jax.device_count()), shared_state),
    )
    
    # Verify trajectories and states
    assert trajectories.action.shape[0] == jax.device_count()
    assert isinstance(next_env_states, RolloutEnvState)
    
    # Check that each device got the expected actions
    for i in range(jax.device_count()):
        assert jnp.allclose(trajectories.action[i], jnp.ones((1,)))


def test_vmap_within_pmap_step_engine(simple_model, rng):
    """Test that vmap within pmap works correctly on step_engine."""
    env_per_device = 2
    num_devices = jax.device_count()
    total_envs = env_per_device * num_devices
    
    config = TestRLConfig(num_envs=total_envs)
    task = TestRLTask(config)
    
    # Create MJX model
    mjx_model = task.get_mjx_model(simple_model)
    
    # Create constants
    dummy_model = DummyModel()
    model_arrs, model_statics = eqx.partition(dummy_model, task.model_partition_fn)
    
    constants = RolloutConstants(
        model_statics=(model_statics,),
        engine=task.get_engine(mjx_model),
        observations=tuple(task.get_observations(mjx_model)),
        commands=tuple(task.get_commands(mjx_model)),
        rewards=tuple(task.get_rewards(mjx_model)),
        terminations=tuple(task.get_terminations(mjx_model)),
        curriculum=task.get_curriculum(mjx_model),
        argmax_action=True,
        aux_constants=xax.FrozenDict({}),
    )
    
    # Create env states - batched for each device
    rngs = jax.random.split(rng, total_envs)
    rngs = rngs.reshape(num_devices, env_per_device, -1)
    
    def create_env_state(init_rng):
        # Setup environment state
        curriculum_state = constants.curriculum.get_initial_state(init_rng)
        physics_state = constants.engine.reset(mjx_model, curriculum_state.level, init_rng)
        
        randomizations = get_physics_randomizers(mjx_model, task.get_physics_randomizers(mjx_model), init_rng)
        commands = get_initial_commands(init_rng, physics_state.data, constants.commands, curriculum_state.level)
        obs_carry = get_initial_obs_carry(init_rng, physics_state, constants.observations)
        reward_carry = get_initial_reward_carry(init_rng, constants.rewards)
        model_carry = task.get_initial_model_carry(init_rng)
        
        return RolloutEnvState(
            commands=commands,
            physics_state=physics_state,
            randomization_dict=randomizations,
            model_carry=model_carry,
            reward_carry=reward_carry,
            obs_carry=obs_carry,
            curriculum_state=curriculum_state,
            rng=init_rng,
        )
    
    # Create vectorized environment states for each device
    vmap_create_env_state = jax.vmap(create_env_state)
    pmap_vmap_create_env_state = jax.pmap(vmap_create_env_state)
    env_states = pmap_vmap_create_env_state(rngs)
    
    # Create shared state (replicated across devices)
    shared_state = RolloutSharedState(
        physics_model=mjx_model,
        model_arrs=(model_arrs,),
        aux_values=xax.FrozenDict({}),
    )
    
    # Create vmapped and pmapped step_engine
    def step_wrapper(env_state, shared_state):
        return task.step_engine(
            constants=constants,
            env_states=env_state,
            shared_state=shared_state,
        )
    
    vmap_step_engine = jax.vmap(step_wrapper)
    pmap_vmap_step_engine = jax.pmap(vmap_step_engine)
    
    # Test vmapped and pmapped step_engine
    trajectories, next_env_states = pmap_vmap_step_engine(
        env_states,
        jax.tree_map(lambda x: jnp.array([x] * jax.device_count()), shared_state),
    )
    
    # Verify trajectories and states
    assert trajectories.action.shape == (num_devices, env_per_device, 3)
    assert isinstance(next_env_states, RolloutEnvState)
    
    # Check that each environment got the expected actions
    for i in range(num_devices):
        for j in range(env_per_device):
            assert jnp.allclose(trajectories.action[i, j], jnp.ones((1,)))