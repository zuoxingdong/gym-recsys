from collections import deque

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
import numpy as np
import scipy
import scipy.special
import gym
import gym.spaces as spaces


class SlateSpace(spaces.MultiDiscrete):
    def __init__(self, nvec):
        assert np.unique(nvec).size == 1, 'each slate position should allow all available items to display.'
        assert len(nvec) <= nvec[0], f'slate size ({len(nvec)}) should be no larger than the number of items ({nvec[0]}).'
        super().__init__(nvec)

    def sample(self):
        # since a slate is a permutation over items with a cut-off
        # we implemented by using numpy for efficiency, avoid for-loop
        return self.np_random.permutation(self.nvec[0])[:len(self.nvec)].astype(self.dtype)

    def sample_batch(self, batch_size):
        # for-loop will be very SLOW!
        # NOTE: we use numpy's `permutation` and `apply_along_axis` to be very efficient!
        n_item = self.nvec[0]
        slate_size = len(self.nvec)

        arr = np.arange(n_item)[None, :]
        arr = np.tile(arr, (batch_size, 1))
        arr = np.apply_along_axis(func1d=self.np_random.permutation, axis=1, arr=arr)
        arr = arr[:, :slate_size]
        return arr

    def contains(self, x):
        is_contained = super().contains(x)
        is_unique = (np.unique(x).size == len(x))
        return is_unique and is_contained

    def __repr__(self):
        return f'SlateSpace({self.nvec})'

    def __eq__(self, other):
        return isinstance(other, SlateSpace) and np.all(self.nvec == other.nvec)


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 3
    }
    reward_range = (-float('inf'), float('inf'))

    def __init__(
        self, user_ids, item_category, item_popularity,
        hist_seq_len, slate_size,
        user_state_model_callback, reward_model_callback,
    ):
        self.user_ids = user_ids
        assert len(item_category) == len(item_popularity)
        item_category = [str(i) for i in item_category]  # enforce str, otherwise visualization won't work well
        self.item_category = item_category
        item_popularity = np.asarray(item_popularity)
        self.scaled_item_popularity = item_popularity/max(item_popularity)
        self.hist_seq_len = hist_seq_len
        self.slate_size = slate_size
        self.user_state_model_callback = user_state_model_callback
        self.reward_model_callback = reward_model_callback

        self.nan_item_id = -1

        self.user_id = None  # enforce calling `env.reset()`
        self.hist_seq = deque([self.nan_item_id]*hist_seq_len, maxlen=hist_seq_len)  # FIFO que for user's historical interactions
        assert len(self.hist_seq) == hist_seq_len

        obs_dim = len(user_state_model_callback(user_ids[0], self.hist_seq))
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        # NOTE: do NOT use `gym.spaces.MultiDiscrete`: it does NOT support unique sampling for slate
        # i.e. a sampled action may contain multiple redundant item in the slate!
        self.action_space = SlateSpace((len(item_category),)*slate_size)

        # some loggings for visualization
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0

        self.viewer = None
        self.fig, self.axes = None, None
        self.seed()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        return self.rng.bit_generator._seed_seq.entropy  # in case `seed=None`, system generated seed will be returned

    def step(self, action):
        assert action in self.action_space
        assert np.unique(action).size == len(action), 'repeated items in slate are not allowed!'
        # append a skip-item at the end of the slate to allow user to skip the slate
        # pre-trained reward model will give a learned reward for skipping
        action = [*action, self.nan_item_id]
        action_item_reward = self.reward_model_callback(self.user_id, self.hist_seq, action)
        assert action_item_reward.ndim == 1 and len(action_item_reward) == len(action)

        # TODO: customize user choice model as input to the environment constructor
        # for the moment, only sampling in proportion to predicted rewards
        choice_dist = scipy.special.softmax(action_item_reward)
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

        self.timestep += 1

        # track interactions for visualization
        self.user_logs.append({
            'timestep': self.timestep,
            'clicked_item_id': clicked_item_id,  # NOTE: include skip activity
            'choice_dist': choice_dist.tolist()
        })
        self.rs_logs.append({
            'timestep': self.timestep,
            'slate': action  # NOTE: include skip pseudo-item
        })

        obs = self._get_obs()
        # Alternative: reward = action_item_reward.min() - 1.*action_item_reward.std()
        reward = action_item_reward[idx]
        if reward <= action_item_reward[-1]:
            reward = 0.
        done = False
        info = {
            'is_click': is_click,
            'clicked_item_id': clicked_item_id,
            'action_item_reward': action_item_reward.tolist(),
            'choice_dist': choice_dist.tolist()
        }
        return obs, reward, done, info

    def _get_obs(self):
        user_state = self.user_state_model_callback(self.user_id, self.hist_seq)  # -> [user_state, ]
        assert user_state in self.observation_space
        return user_state

    def reset(self, **kwargs):
        if kwargs.get('user_id', None) is not None:
            user_id = kwargs['user_id']
            assert user_id in self.user_ids
            self.user_id = user_id
        else:
            self.user_id = self.rng.choice(self.user_ids, size=None)
        self.hist_seq = deque([self.nan_item_id]*self.hist_seq_len, maxlen=self.hist_seq_len)
        assert len(self.hist_seq) == self.hist_seq_len

        # some loggings for visualization
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0

        return self._get_obs()

    def _get_img(self):
        # clear all previous images
        [ax.cla() for ax in self.axes.flatten()]

        # we require strict ordering of the category type in the plot
        # so we use `pd.Categorical` below in `sns.lineplot` to enforce consistent ordering
        categories = np.unique(self.item_category).tolist()
        categories = ['@skip', *categories]
        # enforce str for each category, otherwise `pd.Categorical` breaks with NaN
        categories = [str(c) for c in categories]

        cat_dist_all = pd.Categorical(self.item_category, categories=categories, ordered=True).value_counts()
        cat_dist_all /= cat_dist_all.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        def _barplot_cat_dist_all(cat_dist_all, categories, ax):
            sns.barplot(x=cat_dist_all.index, y=cat_dist_all.values, order=categories, alpha=.3, ax=ax)
            for patch in ax.patches:  # draw dashed edge on top for each true_category, better visual
                x = [patch.get_x(), patch.get_x() + patch.get_width()]
                y = [patch.get_height()]*2
                ax.plot(x, y, ls='--', lw=1.5, c=patch.get_edgecolor(), alpha=1.)

        df_user_logs = pd.DataFrame(self.user_logs).sort_values(by='timestep', ascending=True)
        df_rs_logs = pd.DataFrame(self.rs_logs).sort_values(by='timestep', ascending=True)

        user_click_cat = df_user_logs['clicked_item_id'].apply(
            lambda item_id: str(self.item_category[item_id]) if item_id != self.nan_item_id else '@skip'
        )
        user_click_cat = pd.Categorical(user_click_cat, categories=categories, ordered=True)

        # figure [0, 0]: Overall User Choices
        cat_dist_user = user_click_cat.value_counts()
        cat_dist_user /= cat_dist_user.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        _barplot_cat_dist_all(cat_dist_all, categories, ax=self.axes[0, 0])
        g = sns.barplot(x=cat_dist_user.index, y=cat_dist_user.values, order=categories, alpha=.8, ax=self.axes[0, 0])
        g.set(title='Overall User Choices', ylim=(0., 1.), xlabel='Category', ylabel='Percent')

        # figure [1, 0]: Overall Recommendations
        cat_dist_rs = df_rs_logs.explode('slate')
        cat_dist_rs = cat_dist_rs[cat_dist_rs['slate'] != self.nan_item_id]  # remove skip pseudo-item in slate for visualization
        cat_dist_rs = cat_dist_rs['slate'].apply(
            lambda item_id: str(self.item_category[item_id])
        )
        cat_dist_rs = pd.Categorical(cat_dist_rs, categories=categories, ordered=True).value_counts()
        cat_dist_rs /= cat_dist_rs.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        _barplot_cat_dist_all(cat_dist_all, categories, ax=self.axes[1, 0])
        g = sns.barplot(x=cat_dist_rs.index, y=cat_dist_rs.values, order=categories, alpha=.8, ax=self.axes[1, 0])
        g.set(title='Overall Recommendations', ylim=(0., 1.), xlabel='Category', ylabel='Percent')

        # figure [0, 1]: Sequential User Choices
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=user_click_cat, 
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[0, 1]
        )
        g.set(  # gym animation wrapper `Monitor` requires both `yticks` and `yticklabels`
            title='Sequential User Choices', yticks=range(len(categories)), yticklabels=categories,
            xlabel='Timestep', ylabel='Category'
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        # figure [1, 1]: Intra-Slate Diversity (Shannon)
        rs_diversity = df_rs_logs['slate'].apply(lambda slate: list(filter(lambda x: x != self.nan_item_id, slate)))
        rs_diversity = rs_diversity.apply(
            lambda slate: [str(self.item_category[item_id]) for item_id in slate]
        )
        _categories_wo_skip = list(filter(lambda c: c != '@skip', categories))
        rs_diversity = rs_diversity.apply(lambda slate: pd.Categorical(slate, categories=_categories_wo_skip, ordered=True))
        rs_diversity = rs_diversity.apply(lambda slate: slate.value_counts().values)
        rs_diversity = rs_diversity.apply(lambda slate: slate/slate.sum())
        rs_diversity = rs_diversity.apply(lambda slate: scipy.stats.entropy(slate, base=len(slate)))
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=rs_diversity,
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[1, 1]
        )
        g.set(
            title='Intra-Slate Diversity (Shannon)',
            xlabel='Timestep', ylabel='Shannon Entropy',
            ylim=(0., 1.)
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        # figure [0, 2]: User Choice Distribution
        # make sure the skip pesudo-item is located in the final position
        assert df_rs_logs['slate'].tail(1).item()[-1] == self.nan_item_id
        choice_dist = df_user_logs['choice_dist'].tail(1).item()
        slate_position = list(range(1, self.slate_size+1+1))  # add one more: for skip pseudo-item
        slate_position = [str(i) for i in slate_position]
        slate_position[-1] = '@skip'
        df = pd.DataFrame({'slate_pos': slate_position, 'click_prob': choice_dist})
        g = sns.barplot(
            x='slate_pos', y='click_prob', 
            order=slate_position, alpha=.8, color='b', data=df,
            ax=self.axes[0, 2]
        )
        g.set(title='User Choice Distribution', xlabel='Slate Position', ylabel='Click Probability')

        # figure [1, 2]: Expected Popularity Complement (EPC)
        # EPC: measures the ability to recommend long-tail items in top positions
        # formula: Eq. (7) in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1089.1342&rep=rep1&type=pdf
        slate_epc = df_rs_logs['slate'].apply(lambda slate: list(filter(lambda x: x != self.nan_item_id, slate)))
        _rank_discount = np.log2(np.arange(1, self.slate_size+1) + 1)
        slate_epc = slate_epc.apply(
            lambda slate: np.asarray([1. - self.scaled_item_popularity[item_id] for item_id in slate])/_rank_discount
        )
        slate_epc = slate_epc.apply(
            lambda slate: np.sum(slate)/np.sum(1./_rank_discount)
        )
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=slate_epc,
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[1, 2]
        )
        g.set(
            title='Expected Popularity Complement (EPC)',
            xlabel='Timestep', ylabel='EPC',
            ylim=(0., 1.)
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        self.fig.suptitle(f'User ID: {self.user_id}, Time step: {self.timestep}', y=1.0, size='x-large')
        self.fig.tight_layout()

        self.fig.canvas.draw()
        img = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        img = np.asarray(img)
        return img

    def render(self, mode='human', **kwargs):
        if self.fig is None and self.axes is None:
            self.fig, self.axes = plt.subplots(2, 3, figsize=(3*2*6, 2*2*4))
            sns.set()
        if self.timestep == 0:  # gym Monitor may call `render` at very first step, so return empty image
            self.fig.canvas.draw()
            img = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
            img = np.asarray(img)
        else:
            img = self._get_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                maxwidth = kwargs.get('maxwidth', int(4*500))
                self.viewer = SimpleImageViewer(maxwidth=maxwidth)                
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            plt.close('all')  # close all with matplotlib, free memory
            self.fig = None
            self.axes = None
