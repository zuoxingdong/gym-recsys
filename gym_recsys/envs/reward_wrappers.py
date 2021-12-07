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
            rewards = np.asarray(info['rewards'])
            reward = rewards.min() - 1.*rewards.std()
        return observation, reward, done, info


class PreferenceRewardStochastic(Wrapper):
    def step(self, action):
        observation, _, done, info = self.env.step(action)
        
        rewards = np.asarray(info['rewards'])
        skip_reward = rewards[-1]
        rewards -= skip_reward
        rewards = np.maximum(0, rewards)

        idx = info['clicked_item_idx']
        reward = rewards[idx]
        return observation, reward, done, info


class PreferenceRewardDeterministic(Wrapper):
    def step(self, action):
        observation, _, done, info = self.env.step(action)
        
        rewards = np.asarray(info['rewards'])
        skip_reward = rewards[-1]
        rewards -= skip_reward
        rewards = np.maximum(0, rewards)

        reward = rewards.max()
        return observation, reward, done, info
