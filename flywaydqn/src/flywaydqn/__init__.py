"""
FlywayDQN: RL controller for 5G massive-MIMO beam and power selection under
precautionarily-managed RF exposure over migratory bird flocks.
"""

__version__: str = "1.0.0"

__all__ = ["__version__"]


def get_project_name() -> str:
    """Public, typed entrypoint useful for health checks and smoke tests."""
    return "flywaydqn"
