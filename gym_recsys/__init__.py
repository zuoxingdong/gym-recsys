from gym_recsys.envs import SlateSpace
from gym_recsys.envs import Env

from gym.envs.registration import register


ts = [10, 50, 100, 200, 500, 1000]
for t in ts:
    register(
        id=f'RecSys-t{t}-v0',
        entry_point='gym_recsys.envs:Env',
        max_episode_steps=t
    )
