#!/usr/bin/env python3
"""Test script for distributed training functionality."""

import os
import jax
import jax.numpy as jnp
import numpy as np

# Set multi-device environment
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

# Import after setting environment variable
import ksim
import examples.walking

def test_device_detection():
    """Test that multiple devices are detected correctly."""
    devices = jax.devices()
    print(f"JAX devices detected: {devices}")
    print(f"Number of devices: {len(devices)}")
    return len(devices) > 1

def test_distributed_task_creation():
    """Test that distributed task classes can be created."""
    config = examples.walking.HumanoidWalkingTaskConfig(
        num_envs=8,
        batch_size=4,
        use_distributed=True,
        max_steps=1,  # Very short test
    )
    
    print(f"Config use_distributed: {config.use_distributed}")
    
    # Test task class selection
    if config.use_distributed:
        task_class = examples.walking.HumanoidWalkingDistributedTask
    else:
        task_class = examples.walking.HumanoidWalkingTask
    
    print(f"Selected task class: {task_class.__name__}")
    print(f"Task class MRO: {[cls.__name__ for cls in task_class.__mro__][:6]}")
    
    return task_class

def test_pmap_functionality():
    """Test basic pmap functionality."""
    devices = jax.devices()
    if len(devices) < 2:
        print("Skipping pmap test - need at least 2 devices")
        return True
    
    def simple_computation(x):
        return x * 2 + jax.lax.pmean(x, axis_name='i')
    
    pmapped_computation = jax.pmap(simple_computation, axis_name='i')
    
    # Create test data
    test_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Shape: (2, 2)
    
    try:
        result = pmapped_computation(test_data)
        print(f"Pmap test successful. Input: {test_data}, Output: {result}")
        return True
    except Exception as e:
        print(f"Pmap test failed: {e}")
        return False

def test_distributed_classes():
    """Test that distributed classes are properly implemented."""
    try:
        # Test DistributedRLTask
        distributed_rl = ksim.DistributedRLTask
        print(f"DistributedRLTask available: {distributed_rl}")
        
        # Test DistributedPPOTask  
        distributed_ppo = ksim.DistributedPPOTask
        print(f"DistributedPPOTask available: {distributed_ppo}")
        
        # Check methods exist
        required_methods = ['num_devices', 'use_distributed_training', '_rl_train_loop_step']
        for method in required_methods:
            if hasattr(distributed_rl, method):
                print(f"✓ {method} method found")
            else:
                print(f"✗ {method} method missing")
                return False
        
        return True
    except Exception as e:
        print(f"Distributed classes test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing KSIM Distributed Training ===\n")
    
    print("1. Testing device detection...")
    multi_device = test_device_detection()
    print(f"Multi-device available: {multi_device}\n")
    
    print("2. Testing distributed task creation...")
    try:
        task_class = test_distributed_task_creation()
        print("✓ Distributed task creation successful\n")
    except Exception as e:
        print(f"✗ Distributed task creation failed: {e}\n")
        return False
    
    print("3. Testing pmap functionality...")
    pmap_success = test_pmap_functionality()
    print(f"Pmap test result: {pmap_success}\n")
    
    print("4. Testing distributed classes...")
    classes_success = test_distributed_classes()
    print(f"Distributed classes test result: {classes_success}\n")
    
    # Summary
    all_tests_passed = multi_device and pmap_success and classes_success
    print("=== Test Summary ===")
    print(f"Multi-device detection: {'✓' if multi_device else '✗'}")
    print(f"Pmap functionality: {'✓' if pmap_success else '✗'}")
    print(f"Distributed classes: {'✓' if classes_success else '✗'}")
    print(f"Overall result: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)