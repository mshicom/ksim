"""Test multi-device functionality with proper XLA_FLAGS setup."""

import os
import subprocess
import sys

def run_multidevice_test():
    """Run a test with multiple simulated devices."""
    
    # Set XLA flags before importing JAX
    env = os.environ.copy()
    env['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
    
    test_code = '''
import jax
import jax.numpy as jnp

print("🔍 Multi-Device Test Results:")
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Test distributed utilities
from ksim.task.distributed_rl import get_device_info, replicate_state, shard_environments

process_count, process_id, device_count = get_device_info()
print(f"Device info: {device_count} devices, process {process_id}/{process_count}")

# Test state replication across 2 devices
test_state = {"param": jnp.array([1.0, 2.0, 3.0])}
replicated = replicate_state(test_state, device_count)
print(f"✓ State replication: {test_state['param'].shape} -> {replicated['param'].shape}")

# Test environment sharding across 2 devices
env_state = {"obs": jnp.ones((8, 10))}  # 8 total envs
envs_per_device = 8 // device_count
sharded = shard_environments(env_state, device_count, envs_per_device)
print(f"✓ Environment sharding: {env_state['obs'].shape} -> {sharded['obs'].shape}")

# Test PMean operation
def test_pmean(x):
    return jax.lax.pmean(x, axis_name='device')

pmapped_test = jax.pmap(test_pmean, axis_name='device')
test_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Different per device
result = pmapped_test(test_data)
print(f"✓ PMean test: {test_data} -> {result}")

# Test distributed task setup
from examples.walking_distributed import HumanoidWalkingDistributedConfig, HumanoidWalkingDistributedTask

config = HumanoidWalkingDistributedConfig(
    num_envs=8,  # Will be 4 per device
    batch_size=4,
    rollout_length_seconds=1.0,
    dt=0.01,
    ctrl_dt=0.05,
)

task = HumanoidWalkingDistributedTask(config)
task.verify_distributed_setup()

print("✅ Multi-device test completed successfully!")
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("🔧 Multi-Device Test Output:")
        print("=" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ Multi-device test passed!")
        else:
            print("❌ Multi-device test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🚀 Testing Multi-Device Setup")
    print("=" * 50)
    success = run_multidevice_test()
    
    if success:
        print("\n🎉 All multi-device tests passed!")
        print("\n📝 Usage Instructions:")
        print("To use KSIM distributed training:")
        print("1. Set XLA_FLAGS to specify device count:")
        print("   export XLA_FLAGS='--xla_force_host_platform_device_count=2'")
        print("2. Run your distributed training script:")
        print("   python -m examples.walking_distributed")
        print("3. Ensure num_envs is divisible by device count")
        print("4. The framework will automatically:")
        print("   - Detect available devices") 
        print("   - Shard environments across devices")
        print("   - Replicate model parameters")
        print("   - Synchronize gradients with pmean")
    else:
        print("\n❌ Some tests failed - check implementation")
        sys.exit(1)