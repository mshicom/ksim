# Multi-GPU Training in Ksim

This document describes how to use the multi-GPU training capabilities in Ksim.

## Overview

Ksim now supports distributed training across multiple GPUs/TPUs using JAX's `pmap` functionality. The implementation:

- **Uses all available devices by default**
- **Shards environments across devices** for parallel experience collection
- **Synchronizes gradients across devices** using `pmean`
- **Minimizes cross-device communication** by keeping environment state local
- **Maintains compatibility** with existing single-device code

## Quick Start

### 1. Basic Usage

```python
from ksim.task.distributed_ppo import DistributedPPOConfig, DistributedPPOTask
from examples.walking_distributed import HumanoidWalkingDistributedConfig, HumanoidWalkingDistributedTask

# Create distributed configuration
config = HumanoidWalkingDistributedConfig(
    num_envs=16,        # Total environments (will be sharded across devices)
    batch_size=8,       # Batch size for updates
    rollout_length_seconds=10.0,
    pmap_axis_name="device",
    grad_sync_period=1,  # Sync gradients every N updates
)

# Create and run distributed task
task = HumanoidWalkingDistributedTask(config)
task.verify_distributed_setup()  # Verify configuration
task.run_distributed_training()  # Start training
```

### 2. Command Line Usage

```bash
# For testing with simulated devices
XLA_FLAGS='--xla_force_host_platform_device_count=2' \
python -m examples.walking_distributed num_envs=8 batch_size=4

# For actual multi-GPU training (will auto-detect available devices)
python -m examples.walking_distributed num_envs=32 batch_size=16
```

## Architecture

### Device Management

```python
from ksim.task.distributed_rl import get_device_info

process_count, process_id, local_device_count = get_device_info()
print(f"Running on {local_device_count} devices")
```

### Environment Sharding

Environments are automatically sharded across devices:

```
Total environments: 16
Device 0: environments 0-7
Device 1: environments 8-15
```

### Gradient Synchronization

Gradients are synchronized using `pmean`:

```python
# Local gradient computation on each device
grads = compute_gradients_locally(model, trajectories)

# Synchronize across devices  
synced_grads = jax.lax.pmean(grads, axis_name="device")

# Apply synchronized gradients
updated_model = apply_gradients(model, synced_grads)
```

## Configuration

### DistributedRLConfig

```python
@dataclass
class DistributedRLConfig(RLConfig):
    pmap_axis_name: str = "i"           # Axis name for pmap operations
    grad_sync_period: int = 1           # Sync gradients every N updates
```

### DistributedPPOConfig

Extends `DistributedRLConfig` with all PPO-specific parameters.

## Requirements

### Environment Constraints

- `num_envs` must be divisible by the number of devices
- `batch_size * num_minibatches` should be ≤ `num_envs`

### Device Requirements

- JAX with GPU/TPU support for real multi-device training
- For testing: Use `XLA_FLAGS='--xla_force_host_platform_device_count=N'`

## Testing

Run the included tests to verify your setup:

```bash
# Test basic functionality
python test_distributed.py

# Test multi-device functionality  
python test_multidevice.py
```

## Example Implementation

See `examples/walking_distributed.py` for a complete example of distributed humanoid walking training.

## Performance Considerations

1. **Environment Distribution**: Each device processes its own subset of environments
2. **Gradient Synchronization**: Only gradients are synchronized, not full model state
3. **Communication Overhead**: Minimal - only during gradient sync phase
4. **Memory Usage**: Model parameters are replicated across devices
5. **Scalability**: Linear speedup expected with number of devices

## Troubleshooting

### Common Issues

1. **"num_envs must be divisible by num_devices"**
   - Ensure your environment count is evenly divisible by device count

2. **"compiling computation that requires N devices, but only M are available"**
   - Check that you have enough devices or set XLA_FLAGS for testing

3. **Memory errors**
   - Reduce batch size or number of environments per device

### Debugging

```python
# Check device configuration
task.verify_distributed_setup()

# Monitor device usage
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
```

## Advanced Usage

### Custom Distributed Tasks

To create your own distributed task, extend `DistributedPPOTask`:

```python
class MyDistributedTask(DistributedPPOTask):
    def distributed_rollout(self, model, constants, shared_state, env_state, rng):
        # Implement distributed rollout logic
        pass
        
    def distributed_update_model(self, model, trajectories, rewards, rng):
        # Implement distributed model update with gradient sync
        pass
```

### Custom Gradient Synchronization

```python
# Sync every N steps instead of every step
if step % self.config.grad_sync_period == 0:
    synced_grads = jax.lax.pmean(grads, axis_name=self.config.pmap_axis_name)
else:
    synced_grads = grads  # Use local gradients
```