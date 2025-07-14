# Implementing Multi-GPU Training in Ksim

This document provides high-level instructions for implementing efficient multi-GPU training in the Ksim reinforcement learning framework.

## Overview

We want to adapt Ksim's existing RL framework to efficiently utilize multiple GPUs/TPUs for training. The implementation should:
1. Use all available devices by default
2. Shard environments across devices 
3. Synchronize gradients across devices
4. Minimize cross-device communication
5. Maintain compatibility with existing code

## Implementation Guidelines

### 1. Create New Configuration Class

Extend the existing `PPOConfig` to support distributed training:

```python
@dataclass
class DistributedRLConfig(PPOConfig):
    """Configuration for distributed RL training."""
    
    pmap_axis_name: str = xax.field(
        value="i",
        help="Axis name for pmapping operations."
    )
    
    grad_sync_period: int = xax.field(
        value=1,
        help="Synchronize gradients across devices every N updates."
    )
```

### 2. Device Management Utilities
Implement utilities to handle device information and ensure proper distribution of work:
```python
def get_device_info():
    """Gets information about available devices."""
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    return process_count, process_id, local_device_count
```

### 3. State Management
Implement functions to properly handle state replication and sharding:
* Replicate shared state (model parameters, optimizer state) across devices
* Shard environment states across devices
* Be careful when reshaping arrays to account for 0-sized dimensions:
```python
# IMPORTANT: When reshaping arrays, use this pattern to avoid errors on 0-sized axes
reshaped_x = jax.tree.map(
    lambda x: x.reshape(num_devices, num_envs_per_device, *x.shape[1:]) 
    if hasattr(x, "shape") else x,
    x
)
```

### 4. Core Distributed Training Functions
Implement pmapped versions of the core training functions:
* Distributed rollout function
* Distributed model update with gradient synchronization
* Distributed training step

### 5. Gradient Synchronization
Implement efficient gradient synchronization:
```python
# Synchronize gradients across devices
grads = jax.lax.pmean(grads, axis_name=config.pmap_axis_name)
```

### 6. Training Loop
Create a distributed version of the training loop that:
1. Uses jax.pmap to parallelize computation across devices
2. Ensures proper distribution of environments
3. Periodically verifies state consistency across devices
4. Efficiently logs metrics and checkpoints

## Testing Your Implementation
Use this command for testing on simulated devices:
```bash
XLA_FLAGS='--xla_force_host_platform_device_count=2' python -m examples.walking_distributed num_steps=1000 num_envs=8 batch_size=4
```
Ensure:
1. The batch size times the number of minibatches is divisible by the number of environments
2. The number of environments is divisible by the number of devices
3. Check for any 0-sized dimension errors

## Implementation Files
Create or modify the following files:
`ksim/task/distributed_rl.py` - Core distributed RL functionality
`ksim/task/distributed_ppo.py` - PPO-specific distributed training logic
`examples/walking_distributed.py` - Example implementation using the humanoid walking task

## Code Structure
* Add a new DistributedRLTask class that extends RLTask
* Add a new DistributedPPOTask class that extends PPOTask
* Override key methods to use pmapped versions
* Ensure backward compatibility where possible
* Add concise docstrings and proper type hints

## Performance Considerations
* Minimize cross-device synchronization
* Use pmean for gradient aggregation
* Ensure device-local operations are maximized
* Avoid unnecessary data transfers between host and devices

## Debugging Tips
1. Use jax.debug.print to help debug distributed training issues
2. Implement state verification to catch divergences early
3. Start with small-scale experiments (few environments, short training runs)
4. Compare results to single-device training to verify correctness