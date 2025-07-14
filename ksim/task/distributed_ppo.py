"""Distributed PPO task implementation using JAX pmap for multi-device training."""

__all__ = [
    "DistributedPPOConfig",
    "DistributedPPOTask",
]

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.distributed_rl import DistributedRLConfig, DistributedRLTask, replicate_state, shard_environments
from ksim.task.ppo import PPOConfig, PPOInputs, PPOTask, PPOVariables, compute_ppo_inputs, compute_ppo_loss
from ksim.types import Trajectory

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class DistributedPPOConfig(DistributedRLConfig, PPOConfig):
    """Configuration for distributed PPO training."""
    pass


Config = TypeVar("Config", bound=DistributedPPOConfig)


class DistributedPPOTask(DistributedRLTask[Config], PPOTask[Config], Generic[Config], ABC):
    """Distributed PPO task using JAX pmap for multi-device training."""

    def __init__(self, config: Config) -> None:
        # Initialize both parent classes
        DistributedRLTask.__init__(self, config)
        # Note: PPOTask inherits from RLTask, so we don't call PPOTask.__init__ separately

    @functools.partial(
        jax.pmap,
        axis_name="device",
        static_broadcasted_argnums=(0,)
    )
    def _distributed_rollout(
        self,
        model: PyTree,
        env_state: PyTree,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Trajectory, PyTree, PyTree]:
        """Distributed rollout function that runs on each device.

        Args:
            model: The model to use for rollout.
            env_state: Environment state for this device.
            model_carry: Model carry state.
            rng: Random number generator key.

        Returns:
            Tuple of (trajectory, updated_env_state, updated_model_carry).
        """
        # This calls the base rollout function on each device's subset of environments
        return self._rollout(model, env_state, model_carry, rng)

    @functools.partial(
        jax.pmap,
        axis_name="device",
        static_broadcasted_argnums=(0,)
    )
    def _distributed_compute_ppo_inputs(
        self,
        trajectory: Trajectory,
    ) -> PPOInputs:
        """Distributed computation of PPO inputs.

        Args:
            trajectory: The trajectory to compute inputs for.

        Returns:
            The PPO inputs.
        """
        return compute_ppo_inputs(
            values_t=trajectory.value,
            rewards_t=trajectory.reward.total,
            dones_t=trajectory.done,
            successes_t=trajectory.success,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
            normalize_advantages=self.config.normalize_advantages,
            monte_carlo_returns=self.config.monte_carlo_returns,
        )

    @functools.partial(
        jax.pmap,
        axis_name="device",
        static_broadcasted_argnums=(0,)
    )
    def _distributed_get_ppo_variables(
        self,
        model: PyTree,
        trajectory: Trajectory,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PPOVariables, PyTree]:
        """Distributed computation of PPO variables.

        Args:
            model: The model.
            trajectory: The trajectory.
            model_carry: Model carry state.
            rng: Random number generator key.

        Returns:
            Tuple of (PPO variables, updated model carry).
        """
        return self.get_ppo_variables(model, trajectory, model_carry, rng)

    @functools.partial(
        jax.pmap,
        axis_name="device",
        static_broadcasted_argnums=(0,)
    )
    def _distributed_compute_loss_and_update(
        self,
        model: PyTree,
        opt_state: PyTree,
        trajectory: Trajectory,
        ppo_inputs: PPOInputs,
        on_policy_variables: PPOVariables,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, PyTree, dict[str, Array], PPOVariables]:
        """Distributed model update with gradient synchronization.

        Args:
            model: The current model.
            opt_state: Optimizer state.
            trajectory: The trajectory.
            ppo_inputs: PPO inputs.
            on_policy_variables: On-policy variables.
            model_carry: Model carry state.
            rng: Random number generator key.

        Returns:
            Tuple of (updated_model, updated_opt_state, losses, off_policy_variables).
        """
        # Get off-policy variables
        off_policy_variables, _ = self.get_ppo_variables(model, trajectory, model_carry, rng)

        # Compute losses
        losses_t = compute_ppo_loss(
            on_policy_variables=on_policy_variables,
            off_policy_variables=off_policy_variables,
            ppo_inputs=ppo_inputs,
            clip_param=self.config.clip_param,
            value_loss_coef=self.config.value_loss_coef,
            use_clipped_value_loss=self.config.use_clipped_value_loss,
            entropy_coef=self.config.entropy_coef,
            kl_coef=self.config.kl_coef,
            log_clip_value=self.config.log_clip_value,
        )

        # Compute total loss and gradients
        def compute_total_loss(model: PyTree) -> Array:
            off_policy_vars, _ = self.get_ppo_variables(model, trajectory, model_carry, rng)
            loss_dict = compute_ppo_loss(
                on_policy_variables=on_policy_variables,
                off_policy_variables=off_policy_vars,
                ppo_inputs=ppo_inputs,
                clip_param=self.config.clip_param,
                value_loss_coef=self.config.value_loss_coef,
                use_clipped_value_loss=self.config.use_clipped_value_loss,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                log_clip_value=self.config.log_clip_value,
            )
            return jax.tree.reduce(jnp.add, [loss.mean() for loss in loss_dict.values()])

        # Compute gradients
        loss, grads = jax.value_and_grad(compute_total_loss)(model)

        # Synchronize gradients across devices using pmean
        grads = jax.lax.pmean(grads, axis_name=self.config.pmap_axis_name)

        # Update model using optimizer
        optimizer = self.get_optimizer()
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)

        return new_model, new_opt_state, losses_t, off_policy_variables

    def run_distributed_training(self) -> None:
        """Runs the distributed training loop."""
        logger.info("Starting distributed training...")

        # Verify setup
        self.verify_distributed_setup()

        if self.local_device_count == 1:
            logger.warning("Running with single device - falling back to regular training")
            # For single device, we can still use the distributed code but it won't have any benefit

        # Initialize model and optimizer
        rng = jax.random.PRNGKey(42)  # Use default seed
        rng, model_rng = jax.random.split(rng)

        # Get initial model
        model = self._get_initial_model(model_rng)

        # Initialize optimizer state
        optimizer = self.get_optimizer()
        opt_state = optimizer.init(model)

        # Replicate model and optimizer state across devices
        model = replicate_state(model)
        opt_state = replicate_state(opt_state)

        # Initialize environment states and shard across devices
        rng, env_rng = jax.random.split(rng)
        env_state = self._get_initial_env_state(env_rng)
        env_state = shard_environments(env_state, self.local_device_count, self.config.num_envs)

        # Initialize model carry state
        model_carry = self._get_initial_model_carry()
        model_carry = replicate_state(model_carry)

        logger.info("Distributed training setup complete. Beginning training loop...")

        # Training loop would continue here with pmapped operations
        # This is a placeholder for the full training implementation

    @abstractmethod
    def _get_initial_model(self, rng: PRNGKeyArray) -> PyTree:
        """Gets the initial model.

        Args:
            rng: Random number generator key.

        Returns:
            The initial model.
        """

    @abstractmethod
    def _get_initial_env_state(self, rng: PRNGKeyArray) -> PyTree:
        """Gets the initial environment state.

        Args:
            rng: Random number generator key.

        Returns:
            The initial environment state.
        """

    @abstractmethod
    def _get_initial_model_carry(self) -> PyTree:
        """Gets the initial model carry state.

        Returns:
            The initial model carry state.
        """

    @abstractmethod
    def _rollout(
        self,
        model: PyTree,
        env_state: PyTree,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Trajectory, PyTree, PyTree]:
        """Performs a rollout.

        Args:
            model: The model to use.
            env_state: Environment state.
            model_carry: Model carry state.
            rng: Random number generator key.

        Returns:
            Tuple of (trajectory, updated_env_state, updated_model_carry).
        """
