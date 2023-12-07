from gym.envs.registration import register

register(
    id = "grid_map_env/GridMapEnv-v0",
    entry_point = "grid_map_env.classes:GridMapEnvCompile",
)