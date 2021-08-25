import numpy as np
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
