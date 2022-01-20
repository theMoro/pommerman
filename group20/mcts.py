"""
A nice practical MCTS explanation:
   https://www.youtube.com/watch?v=UXW2yZndl7U
This implementation is based on:
   https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
import math
import random
import numpy as np
from group20 import node as node_py
from group20.game_state import game_state_from_obs

from pommerman import constants


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS:
    def __init__(self, action_space, agent_id, root, policy, rollout_depth=6,
                 exploration_weight=math.sqrt(2), c_puct=5, device='cpu'):
        self.device = device
        self.action_space = action_space
        self._root = root
        self.root_state = root.state
        self.agent_id = agent_id
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.N = 1

        # new vars
        self._policy = policy
        self._c_puct = c_puct

    def choose(self):
        """ Choose the best successor of node. (Choose an action) """
        node = self._root

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        children = node.get_children()
        if len(children) == 0:
            # choose a move randomly, should hopefully never happen
            return self.action_space.get_last_n_samples()

        def score(key):
            n = children[key]
            if n.get_visit_count() == 0:
                return float("-inf")  # avoid unseen moves
            return n.get_total_reward() / n.get_visit_count()  # average reward

        return max(children.keys(), key=score)[self.agent_id]

    def do_rollout(self):
        """ Execute one tree update step: select, expand, simulate, backpropagate """
        node = self._root

        path = self._select(node)
        leaf = path[-1]

        if leaf.obs is None:
            leaf.make_obs()

        action_probs, leaf_value = self._policy(leaf.obs, device=self.device)

        self._expand(leaf, action_probs)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        """ Find an unexplored descendent of node """
        path = []
        while True:
            path.append(node)
            if node.is_terminal() or len(node.get_children()) == 0:
                # node is either unexplored or terminal
                return path

            # if there is an unexplored child node left, take it, because it has highest uct value
            unexplored = self.get_unexplored(node)
            if unexplored:
                path.append(unexplored)
                return path

            node = self._uct_select(node)

    def expand_root(self, root_node):
        if root_node.obs is None:
            root_node.make_obs()

        action_probs, leaf_value = self._policy(root_node.obs, device=self.device)
        root_node.find_children(action_probs)

    @staticmethod
    def _expand(node, action_probs):
        """ expand a node if it has been visited before """
        if node.get_visit_count() > 0:
            node.find_children(action_probs)

    def _simulate(self, node):
        """ performs simulation and returns reward from value function """
        depth = 0
        while True:
            if node.is_terminal() or depth >= self.rollout_depth:
                actions, action_probs, state_v = self.get_nn_outputs(node)
                reward = node.reward(self.root_state, state_v)
                return reward

            node = self._find_random_child(node)
            depth += 1

    def _backpropagate(self, path, reward):

        # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            node.incr_visit_count()
            node.incr_reward(reward)

        # increase total number of steps
        self.N += 1

    def _uct_select(self, node):
        """ Select a child of node, balancing exploration & exploitation """

        children = node.get_children().values()

        visit_count = node.get_visit_count()
        if visit_count == 0:
            return self._find_random_child(node)

        log_n_vertex = math.log(visit_count)

        def uct(n):
            q = n.get_total_reward()
            ni = n.get_visit_count()
            if ni == 0:
                return float('inf')
            "Upper confidence bound for trees"
            return q / ni + self.exploration_weight * math.sqrt(
                log_n_vertex / ni
            )

        return max(children, key=uct)

    def get_nn_outputs(self, node):
        if node.obs is None:
            node.make_obs()

        action_probs, state_v = self._policy(node.obs, device=self.device)
        actions, probs = zip(*action_probs)
        return list(actions), list(probs), state_v

    def _find_random_child(self, node):
        actions, probs, _ = self.get_nn_outputs(node)

        pruned_actions = [a for a in actions if not node.prune(a, is_opponent=False)]
        if len(pruned_actions) == 0:
            # should never happen
            print("LOG: mcts.py, line 165: len(pruned_actions) == 0")
            pruned_actions = [constants.Action.Stop.value]

        action_list = [None, None]
        action_list[self.agent_id] = np.random.choice(pruned_actions)
        action_list[1 - self.agent_id] = np.random.choice(node.pruned_opponent_actions)

        sel_actions = (action_list[0], action_list[1])

        if sel_actions in node.children.keys():
            return node.children[sel_actions]
        else:
            idx = actions.index(action_list[self.agent_id])
            prob = probs[idx]
            child = node.forward(sel_actions, prob)
            return child

    def get_unexplored(self, node):
        """ returns a randomly chosen unexplored action pair, or None """
        actions, probs, _ = self.get_nn_outputs(node)

        pruned_actions = [a for a in actions if not node.prune(a, is_opponent=False)]
        if len(pruned_actions) == 0:
            print("LOG: mcts.py, line 188: len(pruned_actions) == 0")
            pruned_actions = [constants.Action.Stop.value]

        action_list = [None, None]
        action_list[self.agent_id] = pruned_actions
        action_list[1 - self.agent_id] = node.pruned_opponent_actions
        action_combos = [(a1, a2) for a1 in action_list[0] for a2 in action_list[1]]

        unexplored_actions = [a for a in action_combos if a not in node.children.keys()]
        if not unexplored_actions:
            return None

        sel_actions = random.choice(unexplored_actions)
        idx = actions.index(sel_actions[self.agent_id])
        prob = probs[idx]
        child = node.forward(sel_actions, prob)
        node.children[sel_actions] = child
        return child

    def update_root(self, obs, agent_id):
        game_state = game_state_from_obs(obs, self.agent_id)
        self._root = node_py.Node(None, game_state, agent_id, 1.0, obs=obs)
        self.expand_root(self._root)

    def get_move_probs(self, temp=1e-3):
        act_visits = [(act[self.agent_id], n.get_visit_count()) for act, n in self._root.children.items()]
        act_visits = np.array(act_visits)

        visits = [0] * 6
        for i in range(6):
            if len(act_visits.shape) == 2:
                visits[i] = sum(act_visits[act_visits[:, 0] == i][:, 1])
            else:
                print("LOG: mcts.py, line 221: act_visits in unusual shape")
                return self.action_space.get_last_n_samples(), [1.0]

        acts, _ = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return [0, 1, 2, 3, 4, 5], act_probs


class MCTSNode(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    """

    @abstractmethod
    def find_children(self, action_prior_probs):
        # expands all children
        pass

    @abstractmethod
    def get_children(self):
        # returns all children
        return list()

    @abstractmethod
    def get_total_reward(self):
        # total reward of a node
        return 0

    @abstractmethod
    def get_visit_count(self):
        # Total number of times visited this node (N)
        return 0

    @abstractmethod
    def incr_visit_count(self):
        return 0

    @abstractmethod
    def is_terminal(self):
        # Returns True if the node has no children
        return True

    @abstractmethod
    def reward(self, root_state, state_v):
        # either reward or in our case the return value of the value function
        return 0
