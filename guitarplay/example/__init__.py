from gymnasium.envs.registration import register
import gymnasium as gym
env_id="GuitarPlay-v0"
if env_id not in gym.envs.registry:
    register(
        id="GuitarPlay-v0",
        entry_point="example.envs:GuitarPlayEnv",
    ) 
else:
    print(f"Environment {env_id} is already registered.")