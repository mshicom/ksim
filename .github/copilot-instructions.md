## Repo Structure
refer to deepwiki on repo `kscalelabs/ksim`

## Development Flow
- test: `XLA_FLAGS='--xla_force_host_platform_device_count=2' python -m examples.walking max_steps=3 rollout_length_seconds=0.1 num_envs=16 batch_size=4`
- static checks: `mkdir -p .mypy_cache; make static-checks`

## Key Guidelines
1. Maintain existing code structure and organization
2. Write unit tests for new functionality. Use table-driven unit tests when possible.
3. Keep the docstring concise.