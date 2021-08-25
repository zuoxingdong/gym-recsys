import numpy as np
from gym import Wrapper


class CTRReward(Wrapper):
    def step(self, action):
        observation, _, done, info = self.env.step(action)
        if info['is_click']:
            reward = 1.
        else:
            reward = 0.
        return observation, reward, done, info
        

class SkipPenaltyReward(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if not info['is_click']:
            action_item_reward = np.asarray(info['action_item_reward'])
            reward = action_item_reward.min() - 1.*action_item_reward.std()
        return observation, reward, done, info


class PreferenceRewardStochastic(Wrapper):
    def step(self, action):
        observation, _, done, info = self.env.step(action)
        
        action_item_reward = np.asarray(info['action_item_reward'])
        skip_reward = action_item_reward[-1]
        action_item_reward -= skip_reward
        action_item_reward = np.maximum(0, action_item_reward)

        idx = info['clicked_item_idx']
        reward = action_item_reward[idx]
        return observation, reward, done, info


class PreferenceRewardDeterministic(Wrapper):
    def step(self, action):
        observation, _, done, info = self.env.step(action)
        
        action_item_reward = np.asarray(info['action_item_reward'])
        skip_reward = action_item_reward[-1]
        action_item_reward -= skip_reward
        action_item_reward = np.maximum(0, action_item_reward)

        reward = action_item_reward.max()
        return observation, reward, done, info
