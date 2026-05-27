import numpy as np


#----------------------------------------------------------------------------#
#                         NARROW REPLAY BUFFER                               #
#----------------------------------------------------------------------------#
class NarrowReplayBuffer:
    """PER buffer for narrow pretrain. Samples are i.i.d. (stateless MLP, no LSTM)."""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha    = alpha
        self._buf     = []
        self._prios   = np.zeros(capacity, dtype=np.float32)
        self._pos     = 0

    def add(self, dx, dy, wide_goal, narrow_goal, reward, log_prob):
        p = (max(reward, 0.0) + 1e-6) ** self.alpha
        if len(self._buf) < self.capacity:
            self._buf.append((dx, dy, wide_goal, narrow_goal, reward, log_prob))
        else:
            self._buf[self._pos] = (dx, dy, wide_goal, narrow_goal, reward, log_prob)
        self._prios[self._pos] = p
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        n       = len(self._buf)
        prios   = self._prios[:n]
        probs   = prios / prios.sum()
        replace = n < batch_size
        idxs    = np.random.choice(n, size=batch_size, replace=replace, p=probs)
        weights = (n * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return [self._buf[i] for i in idxs], weights.astype(np.float32)

    def __len__(self):
        return len(self._buf)


#----------------------------------------------------------------------------#
#                      WORKER EPISODE REPLAY BUFFER                         #
#----------------------------------------------------------------------------#
class WorkerEpisodeReplayBuffer:
    """PER buffer for Worker pretrain. Stores full episodes (GAE requires complete sequences)."""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha    = alpha
        self._buf     = []
        self._prios   = np.zeros(capacity, dtype=np.float32)
        self._pos     = 0

    def add(self, episode, total_reward):
        p = (max(total_reward, 0.0) + 1e-6) ** self.alpha
        if len(self._buf) < self.capacity:
            self._buf.append(episode)
        else:
            self._buf[self._pos] = episode
        self._prios[self._pos] = p
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, n, beta):
        sz    = len(self._buf)
        prios = self._prios[:sz]
        probs = prios / prios.sum()
        idxs  = np.random.choice(sz, size=n, replace=(sz < n), p=probs)
        weights = (sz * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return [self._buf[i] for i in idxs], weights.astype(np.float32)

    def clear(self):
        self._buf   = []
        self._prios = np.zeros(self.capacity, dtype=np.float32)
        self._pos   = 0

    def __len__(self):
        return len(self._buf)
