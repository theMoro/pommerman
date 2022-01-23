from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_probs', 'reward', 'terminal', 'penalty'))


class ReplayMemory(object):
    """
    This class saves transitions that are used for optimization.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def get_last_n_samples(self, n):
        # return the last pushed data samples
        return self.memory[-n:]

    def __len__(self):
        return len(self.memory)
