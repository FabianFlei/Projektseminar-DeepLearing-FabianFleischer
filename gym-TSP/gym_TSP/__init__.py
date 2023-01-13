from gym.envs.registration import register

register(
    id='TSPEnv-v0',
    entry_point='gym_TSP.envs:TSPEnv',
)