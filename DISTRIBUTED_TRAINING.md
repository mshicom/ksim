# Distributed Training Guide for Ksim

This document provides a comprehensive guide for using multi-GPU/TPU training capabilities in the Ksim reinforcement learning framework.

## Quick Start

### Basic Usage

```python
from examples.walking_distributed import HumanoidWalkingDistributedConfig, HumanoidWalkingDistributedTask

# Create distributed configuration
config = HumanoidWalkingDistributedConfig(
    num_envs=16,        # Will be sharded across available devices
    batch_size=8,
    pmap_axis_name="device",
    grad_sync_period=1, # Sync gradients every update
)

# Create and run distributed task
task = HumanoidWalkingDistributedTask(config)
task.verify_distributed_setup()
task.run_distributed_training()
```

### Command Line Usage

```bash
# Production multi-GPU training
python -m examples.walking_distributed num_envs=32 batch_size=16

# Test with simulated devices (useful for development)
XLA_FLAGS='--xla_force_host_platform_device_count=2' \
python -m examples.walking_distributed num_envs=8 batch_size=4

# Single device (automatic fallback)
python -m examples.walking_distributed num_envs=8 batch_size=4
```

## Architecture Overview

### Core Components

1. **`ksim/task/distributed_rl.py`** - Base distributed RL functionality
   - `DistributedRLConfig`: Configuration class with pmap settings
   - `DistributedRLTask`: Base class for distributed training
   - Device management utilities

2. **`ksim/task/distributed_ppo.py`** - PPO-specific distributed training
   - `DistributedPPOTask`: Extends PPOTask with pmapped operations
   - Distributed rollout, loss computation, and model updates
   - Gradient synchronization using `jax.lax.pmean()`

3. **`examples/walking_distributed.py`** - Working example
   - Complete implementation for humanoid walking task
   - Shows integration with existing Ksim components

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Device Training                     │
├─────────────────────────────────────────────────────────────┤
│  Device 0              Device 1              Device N       │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐     │
│  │ Envs    │           │ Envs    │           │ Envs    │     │
│  │ 0-7     │           │ 8-15    │           │ ...     │     │
│  └─────────┘           └─────────┘           └─────────┘     │
│       │                     │                     │         │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐     │
│  │ Local   │           │ Local   │           │ Local   │     │
│  │ Rollout │           │ Rollout │           │ Rollout │     │
│  └─────────┘           └─────────┘           └─────────┘     │
│       │                     │                     │         │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐     │
│  │ Local   │           │ Local   │           │ Local   │     │
│  │ Grads   │           │ Grads   │           │ Grads   │     │
│  └─────────┘           └─────────┘           └─────────┘     │
│       └─────────────────────┼─────────────────────┘         │
│                    ┌─────────────────┐                      │
│                    │ Gradient Sync   │                      │
│                    │ (pmean)         │                      │
│                    └─────────────────┘                      │
│                             │                               │
│                    ┌─────────────────┐                      │
│                    │ Model Update    │                      │
│                    │ (replicated)    │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### State Management

- **Model Parameters**: Replicated across all devices using `replicate_state()`
- **Environment States**: Sharded across devices using `shard_environments()`
- **Gradients**: Computed locally, then synchronized via `jax.lax.pmean()`
- **Optimizer State**: Replicated to maintain consistency

## Configuration Options

### `DistributedRLConfig` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pmap_axis_name` | `"device"` | Axis name for pmapping operations |
| `grad_sync_period` | `1` | Synchronize gradients every N updates |

### Environment Distribution

The framework automatically distributes environments across available devices:

```python
# Example: 16 environments, 2 devices
Total environments: 16
Device 0: environments 0-7  
Device 1: environments 8-15

# Example: 32 environments, 4 devices  
Total environments: 32
Device 0: environments 0-7
Device 1: environments 8-15
Device 2: environments 16-23
Device 3: environments 24-31
```

### Constraints

- `num_envs` must be divisible by number of devices
- `batch_size * num_batches` must equal `num_envs`
- All devices must have the same compute capability

## Implementation Guide

### Creating a Distributed Task

1. **Extend the distributed base class**:
```python
from ksim.task import DistributedPPOTask, DistributedPPOConfig

class MyDistributedTask(DistributedPPOTask[MyConfig]):
    # Implement required abstract methods
    pass
```

2. **Implement required methods**:
```python
def _get_initial_model(self, rng: PRNGKeyArray) -> Model:
    """Create and return the initial model."""
    
def _get_initial_env_state(self, rng: PRNGKeyArray) -> EnvState:
    """Create and return initial environment state."""
    
def _rollout(self, model, env_state, model_carry, rng):
    """Implement rollout logic."""
```

3. **Configure for your use case**:
```python
@dataclass
class MyDistributedConfig(DistributedPPOConfig):
    # Add task-specific configuration
    my_param: float = 1.0
```

### Advanced Usage

#### Custom Gradient Synchronization Period

```python
config = MyDistributedConfig(
    grad_sync_period=4,  # Sync every 4 updates for reduced communication
)
```

#### Device-Specific Environment Counts

```python
# Automatically handled based on available devices
device_count = jax.local_device_count()
envs_per_device = config.num_envs // device_count
```

## Performance Considerations

### Optimization Tips

1. **Environment Count**: Use multiples of device count for optimal load balancing
2. **Batch Size**: Larger batches reduce communication overhead
3. **Gradient Sync**: Consider increasing `grad_sync_period` for communication-bound workloads
4. **Memory**: Each device loads a subset of environments, reducing memory per device

### Expected Performance

- **Linear speedup** with number of devices for compute-bound workloads
- **Communication overhead** only during gradient synchronization
- **Memory efficiency** through environment sharding

### Benchmarking

```python
# Monitor training metrics
task.verify_distributed_setup()  # Shows device/environment distribution
# Training automatically logs per-device metrics
```

## Troubleshooting

### Common Issues

#### "Number of environments not divisible by devices"
```
ValueError: Number of environments (10) must be divisible by number of devices (3)
```
**Solution**: Adjust `num_envs` to be divisible by device count.

#### "Batch size configuration invalid"
```
ValueError: Batch size (8) times number of batches (2) must equal number of environments (10)
```
**Solution**: Ensure `batch_size * num_batches = num_envs`.

#### Out of Memory
**Symptoms**: JAX OOM errors during initialization
**Solutions**: 
- Reduce `num_envs` per device
- Reduce `batch_size`
- Use gradient checkpointing if available

#### Poor Performance
**Symptoms**: Slower than expected scaling
**Solutions**:
- Increase `batch_size` to reduce communication frequency
- Increase `grad_sync_period` for communication-bound workloads
- Check that all devices have similar compute capability

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

task = MyDistributedTask(config)
task.verify_distributed_setup()  # Shows detailed setup information
```

### Single Device Fallback

The framework automatically falls back to single-device training when only one device is available:

```
WARNING: Only 1 device detected. Distributed training will fall back to single-device mode.
```

This ensures backward compatibility with existing code.

## Advanced Topics

### Custom Communication Patterns

For advanced users who need custom gradient synchronization:

```python
class CustomDistributedTask(DistributedPPOTask):
    @functools.partial(jax.pmap, axis_name="device")
    def custom_gradient_sync(self, grads):
        # Custom synchronization logic
        return jax.lax.pmean(grads, axis_name="device")
```

### Multi-Node Training

The current implementation supports single-node multi-GPU training. For multi-node training, additional setup is required:

```bash
# Multi-node training (future enhancement)
# This would require additional configuration and setup
```

### Integration with Existing Code

Distributed training is designed to be a drop-in replacement:

```python
# Original code
task = HumanoidWalkingTask(config)

# Distributed code  
task = HumanoidWalkingDistributedTask(distributed_config)
```

All existing task methods and functionality remain available.

## FAQ

**Q: Does this work with TPUs?**
A: Yes, JAX's pmap works with both GPUs and TPUs.

**Q: Can I mix different device types?**
A: No, all devices should be of the same type and capability.

**Q: What's the minimum number of environments needed?**
A: At least `num_devices * batch_size` environments.

**Q: How do I monitor training progress?**
A: Use the same logging and visualization tools as single-device training.

**Q: Is this compatible with existing checkpoints?**
A: Yes, model checkpoints are compatible between single and distributed training.

## Examples

### Simple 2-GPU Setup
```bash
XLA_FLAGS='--xla_force_host_platform_device_count=2' \
python -m examples.walking_distributed num_envs=8 batch_size=4
```

### Production 8-GPU Setup
```bash
python -m examples.walking_distributed num_envs=64 batch_size=32
```

### Custom Task Implementation
```python
class MyDistributedTask(DistributedPPOTask[MyConfig]):
    def get_rewards(self, physics_model):
        return [MyCustomReward()]
    
    def _get_initial_model(self, rng):
        return MyModel(rng)
```

For more examples and detailed API documentation, see the source code in `ksim/task/distributed_*.py`.