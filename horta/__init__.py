from gymnasium.envs.registration import register, registry
from .env import HortaEnv

if "HortaComunitaria-v0" not in registry:
    register(id="HortaComunitaria-v0", entry_point="horta.env:HortaEnv")

__all__ = ["HortaEnv"]
