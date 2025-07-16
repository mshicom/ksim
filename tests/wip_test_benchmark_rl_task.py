"""Benchmark script for RLTask step_engine with vmap and pmap operations."""

import argparse
import time
from typing import Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xax
from tabulate import tabulate

from ksim.task.rl import RLTask, RolloutConstants, RolloutEnvState, RolloutSharedState
from tests.wip_test_rl_vmap_pmap import TestRLConfig, TestRLTask


def time_function(func: Callable, *args, **kwargs) -> float:
    """Time a function call in milliseconds."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    _ = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
    return (end_time - start_time) * 1000


def benchmark_step_engine(
    num_envs: int,
    batch_size: int,
    rollout_length: float,
    use_pmap: bool,
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
) -> Dict[str, float]:
    """Benchmark step_engine with different configurations."""
    # Configure the task
    config = TestRLConfig(
        num_envs=num_envs,
        batch_size=batch_size,
        rollout_length_seconds=rollout_length,
    )
    task = TestRLTask(config)
    
    # Get model and environment setup
    rng = jax.random.PRNGKey(0)
    mj_model = task.get_mujoco_model()
    mjx_model = task.get_mjx_model(mj_model)
    
    # Create dummy model and get constants
    dummy_model = task.dummy_model
    model_arrs, model_statics = eqx.partition(dummy_model, task.model_partition_fn)
    
    # Setup constants
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
    
    # Measure initialization time
    start_init = time.time()
    
    if use_pmap:
        # We'll use vmap within pmap
        devices = jax.devices()
        num_devices = len(devices)
        envs_per_device = num_envs // num_devices
        
        if num_envs % num_devices != 0:
            print(f"Warning: {num_envs} environments not divisible by {num_devices} devices.")
            num_envs = envs_per_device * num_devices
            print(f"Adjusting to {num_envs} environments.")
        
        # Create RNGs for each device and environment
        rngs = jax.random.split(rng, num_devices * envs_per_device)
        rngs = rngs.reshape(num_devices, envs_per_device, -1)
        
        # Create environment states for each device
        def create_env_states_for_device(device_rngs):
            return jax.vmap(task._get_env_state)(
                rng=device_rngs,
                rollout_constants=constants,
                mj_model=mj_model,
                physics_model=mjx_model,
                randomizers=task.get_physics_randomizers(mjx_model),
            )
        
        env_states = jax.pmap(create_env_states_for_device)(rngs)
        
        # Create shared state (replicated across devices)
        shared_state = RolloutSharedState(
            physics_model=mjx_model,
            model_arrs=(model_arrs,),
            aux_values=xax.FrozenDict({}),
        )
        shared_state_pmap = jax.tree_map(
            lambda x: jnp.array([x] * num_devices),
            shared_state
        )
        
        # Create vmapped and pmapped step_engine
        vmap_step_engine = jax.vmap(
            lambda env_state, shared_state: task.step_engine(
                constants=constants,
                env_states=env_state,
                shared_state=shared_state,
            )
        )
        step_fn = jax.pmap(vmap_step_engine)
        
        def run_step():
            return step_fn(env_states=env_states, shared_state=shared_state_pmap)
        
    else:
        # Just use vmap
        rngs = jax.random.split(rng, num_envs)
        
        # Create environment states
        env_states = jax.vmap(task._get_env_state)(
            rng=rngs,
            rollout_constants=constants,
            mj_model=mj_model,
            physics_model=mjx_model,
            randomizers=task.get_physics_randomizers(mjx_model),
        )
        
        # Create shared state
        shared_state = RolloutSharedState(
            physics_model=mjx_model,
            model_arrs=(model_arrs,),
            aux_values=xax.FrozenDict({}),
        )
        
        # Create vmapped step_engine
        step_fn = jax.vmap(task.step_engine, in_axes=(None, 0, None))
        
        def run_step():
            return step_fn(
                constants=constants,
                env_states=env_states,
                shared_state=shared_state,
            )
    
    init_time = (time.time() - start_init) * 1000
    
    # Warmup
    for _ in range(warmup_steps):
        _ = run_step()
    
    # Benchmark
    step_times = []
    for _ in range(benchmark_steps):
        step_time = time_function(run_step)
        step_times.append(step_time)
    
    # Calculate metrics
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    min_step_time = np.min(step_times)
    max_step_time = np.max(step_times)
    
    env_steps_per_second = num_envs * 1000 / avg_step_time
    
    return {
        "init_time_ms": init_time,
        "avg_step_time_ms": avg_step_time,
        "std_step_time_ms": std_step_time,
        "min_step_time_ms": min_step_time,
        "max_step_time_ms": max_step_time,
        "env_steps_per_second": env_steps_per_second,
        "total_envs": num_envs,
        "parallelism": "pmap+vmap" if use_pmap else "vmap",
    }


def run_benchmarks():
    """Run a series of benchmarks with different configurations."""
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Benchmark configurations
    configs = [
        # Test vmap with increasing number of environments
        {"num_envs": 1, "batch_size": 1, "rollout_length": 0.1, "use_pmap": False},
        {"num_envs": 4, "batch_size": 4, "rollout_length": 0.1, "use_pmap": False},
        {"num_envs": 16, "batch_size": 4, "rollout_length": 0.1, "use_pmap": False},
        {"num_envs": 64, "batch_size": 4, "rollout_length": 0.1, "use_pmap": False},
        {"num_envs": 256, "batch_size": 4, "rollout_length": 0.1, "use_pmap": False},
    ]
    
    # Add pmap configurations if multiple devices available
    if jax.device_count() > 1:
        configs.extend([
            # Test pmap with vmap on multiple devices
            {"num_envs": 4, "batch_size": 2, "rollout_length": 0.1, "use_pmap": True},
            {"num_envs": 16, "batch_size": 4, "rollout_length": 0.1, "use_pmap": True},
            {"num_envs": 64, "batch_size": 4, "rollout_length": 0.1, "use_pmap": True},
            {"num_envs": 256, "batch_size": 4, "rollout_length": 0.1, "use_pmap": True},
        ])
    
    results = []
    for config in configs:
        print(f"\nRunning benchmark with {config}...")
        result = benchmark_step_engine(**config)
        results.append(result)
        print(f"Result: {result}")
    
    # Display results in a table
    table_data = []
    for result in results:
        table_data.append([
            f"{result['parallelism']}",
            f"{result['total_envs']}",
            f"{result['init_time_ms']:.2f}",
            f"{result['avg_step_time_ms']:.2f} ± {result['std_step_time_ms']:.2f}",
            f"{result['env_steps_per_second']:.2f}",
        ])
    
    headers = [
        "Parallelism", 
        "Num Envs", 
        "Init Time (ms)",
        "Step Time (ms)",
        "Env Steps/Sec"
    ]
    
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RLTask step_engine")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--rollout_length", type=float, default=0.1, help="Rollout length in seconds")
    parser.add_argument("--use_pmap", action="store_true", help="Use pmap for parallelization")
    
    args = parser.parse_args()
    
    if args.num_envs != 0 and args.batch_size != 0:
        # Run a single benchmark with provided arguments
        result = benchmark_step_engine(
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            rollout_length=args.rollout_length,
            use_pmap=args.use_pmap,
        )
        print("\nBenchmark Result:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        # Run all benchmarks
        run_benchmarks()