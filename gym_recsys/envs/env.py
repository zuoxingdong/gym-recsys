from collections import deque
import collections.abc

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import gym
import gym.spaces as spaces

from gym_recsys.envs.slate_space import SlateSpace


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 3
    }
    reward_range = (-float('inf'), float('inf'))

    def _check_valid_inputs(
        self, 
        user_ids, 
        hist_len, 
        n_items, 
        slate_size, 
        user_state_callback,
        reward_callback
    ):
        if not isinstance(user_ids, collections.abc.Sequence):
            raise TypeError(f'`user_ids` must be a sequence, got {type(user_ids)}')
        if len(user_ids) == 0:
            raise ValueError(f'`user_ids` must be non-empty')
        if not isinstance(hist_len, int) or hist_len <= 0:
            raise ValueError(f'`hist_len` must be a positive integer, got {hist_len}')
        if not isinstance(n_items, int) or n_items <= 0:
            raise ValueError(f'`n_items` must be a positive integer, got {n_items}')
        if not isinstance(slate_size, int) or slate_size <= 0 or slate_size > n_items:
            raise ValueError(f'`slate_size` must be a positive integer and no greater than `n_items`, got {slate_size}')
        if not isinstance(user_state_callback, collections.abc.Callable):
            raise TypeError(f'`user_state_callback` must a callable, got {type(user_state_callback)}')
        if not isinstance(reward_callback, collections.abc.Callable):
            raise TypeError(f'`reward_callback` must be a callable, got {type(reward_callback)}')
        # TODO: use `inspect.signature` to check arguments of `user_state_callback` and `reward_callback`

    def __init__(
        self,
        user_ids,
        hist_len,
        n_items,
        slate_size,
        user_state_callback,
        reward_callback
    ):
        self._check_valid_inputs(user_ids, hist_len, n_items, slate_size, user_state_callback, reward_callback)
        self.user_ids = user_ids
        self.hist_len = hist_len
        self.n_items = n_items
        self.slate_size = slate_size
        self.user_state_callback = user_state_callback
        self.reward_callback = reward_callback

        self.nan_item_id = -1
        self.user_id = None  # enforce calling `env.reset()`
        self.hist_seq = None
        
        # Define observation and action spaces
        self.observation_space = spaces.MultiDiscrete((n_items,)*hist_len)
        # NOTE: do NOT use `gym.spaces.MultiDiscrete`: it does NOT support unique sampling for slate
        # i.e. a sampled action may contain multiple redundant item in the slate!
        self.action_space = SlateSpace((n_items,)*slate_size)

        # Some meta info
        self.timestep = 0
        self.viewer = None
        self.fig, self.axes = None, None

        self.seed()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        return self.rng.bit_generator._seed_seq.entropy  # in case `seed=None`, system generated seed will be returned

    def _get_obs(self):
        user_state = self.user_state_callback(self.user_id, self.hist_seq)
        return user_state

    def step(self, action):
        assert action in self.action_space
        assert np.unique(action).size == len(action), 'repeated items in slate are not allowed!'
        # append a skip-item at the end of the slate to allow user to skip the slate
        # e.g. pre-trained reward model will give a learned reward for skipping
        action = [*action, self.nan_item_id]
        rewards = self.reward_callback(self.user_id, self.hist_seq, action)
        assert rewards.ndim == 1 and len(rewards) == len(action)

        # TODO: customize user choice model as input to the environment constructor
        # for now: only sampling in proportion to predicted rewards
        choice_dist = scipy.special.softmax(rewards)
        idx = self.rng.choice(len(action), size=None, p=choice_dist)
        clicked_item_id = action[idx]
        is_click = (clicked_item_id != self.nan_item_id)

        # update user state transition
        # NOTE: when user skips, `hist_seq` will not change. 
        # For RL agent training (e.g. DQN), it's important to have exploration!
        # Otherwise, agent might get stuck with suboptimal behavior by repeated observation
        # Also, replay buffer may be dominated by such transitions with identical observations
        if is_click:  # user clicked an item in the slate
            self.hist_seq.append(clicked_item_id)

        # update meta info
        self.timestep += 1

        obs = self._get_obs()
        reward = rewards[idx]
        done = False  # never-ending
        info = {
            'is_click': is_click,
            'clicked_item_idx': idx,
            'clicked_item_id': clicked_item_id,
            'rewards': rewards.tolist(),
            'choice_dist': choice_dist.tolist()
        }
        return obs, reward, done, info

    def reset(self, **kwargs):
        if kwargs.get('user_id', None) is not None:
            user_id = kwargs['user_id']
            assert user_id in self.user_ids
            self.user_id = user_id
        else:
            self.user_id = self.rng.choice(self.user_ids, size=None)
        self.hist_seq = deque([self.nan_item_id]*self.hist_len, maxlen=self.hist_len)  # FIFO que for user's historical interactions
        assert len(self.hist_seq) == self.hist_len

        # some meta info resetting
        self.timestep = 0
        return self._get_obs()

    def render(self, mode='human', **kwargs):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            plt.close('all')  # close all with matplotlib, free memory
            self.viewer = None
            self.fig, self.axes = None, None
