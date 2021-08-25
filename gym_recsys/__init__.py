from gym_recsys.envs import SlateSpace
from gym_recsys.envs import Env
from gym_recsys.envs import CTRReward, SkipPenaltyReward, PreferenceRewardDeterministic, PreferenceRewardStochastic

from gym.envs.registration import register


for t in range(1, 1000+1):
    register(
        id=f'RecSys-t{t}-v0',
        entry_point='gym_recsys.envs:Env',
        max_episode_steps=t
    )
