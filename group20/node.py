import copy
import random
import numpy as np

from pommerman.constants import Action
from pommerman.agents import DummyAgent
from pommerman.forward_model import ForwardModel
from pommerman import constants
from pommerman import characters
from pommerman.constants import Item

from group20.mcts import MCTSNode

ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]


class Node(MCTSNode):
    """A node in the MCTS tree.
        Each node keeps track of its own value Q, prior probability P, and
        its visit-count-adjusted prior score u.
        """

    def get_unexplored(self):
        pass

    def find_random_child(self):
        pass

    def __init__(self, parent, state, agent_id, prior_p, obs=None):
        self._parent = parent
        self.total_reward = 0
        self.visit_count = 0
        # state is a list of: 0. Board, 1. Agents, 2. Bombs, 3. Items, 4. Flames
        self.state = state
        self.agent_id = agent_id

        self.children = dict()

        # new vars
        self.n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

        # board = state[0]
        # agents = state[1]
        # bombs = state[2]
        # flames = state[4]
        #
        # if obs is None:
        #     fm = ForwardModel()
        #     obss = fm.get_observations(board, agents, bombs, flames, False, 10,
        #                                constants.GameType.FFA, 'pommerman.envs.v0:Pomme')
        #     obs = obss[agent_id]
        #
        self.obs = obs

        self.pruned_opponent_actions = [a for a in range(6) if not self.prune(a)]

    def make_obs(self):
        board = self.state[0]
        agents = self.state[1]
        bombs = self.state[2]
        flames = self.state[4]

        fm = ForwardModel()
        obss = fm.get_observations(board, agents, bombs, flames, False, 10,
                                   constants.GameType.FFA, 'pommerman.envs.v0:Pomme')
        self.obs = obss[self.agent_id]

    def get_my_children_actions(self):
        my_actions = []

        for child in self.children.keys():
            my_actions.append(child[0])

        return my_actions

    def prune(self, action, is_opponent=True):
        # TODO: here you can think about more complex stategies to prune moves,
        #   which allows you to create deeper search trees (very important!)
        # remember: two agents -> ids: 0 and 1
        own_agent = self.state[1][self.agent_id]
        opponent_agent = self.state[1][1 - self.agent_id]
        own_position = own_agent.position
        opponent_position = opponent_agent.position
        man_dist = manhattan_dist(own_position, opponent_position)
        if is_opponent and man_dist > 6 and action != Action.Stop.value:
            # we do not model the opponent, if it is more than 6 steps away
            return True

        # a lot of moves (e.g. bumping into a wall or wooden tile) actually result in stop moves
        # we do not have to consider, since they lead to the same result as actually playing a stop move

        if is_opponent:
            if self._is_legal_action(opponent_position, action):
                return False  # not prune actions
        else:
            if self._is_legal_action(own_position, action):
                return False

        return True

    def _is_legal_action(self, position, action):
        """ prune moves that lead to stop move"""
        if action == Action.Stop.value:
            return True
        board = self.state[0]
        bombs = self.state[2]
        bombs = [bomb.position for bomb in bombs]
        row = position[0]
        col = position[1]
        # if it a bomb move, check if there is already a bomb planted on this field
        if action == Action.Bomb.value and (row, col) in bombs:
            return False

        if action == Action.Up.value:
            row -= 1
        elif action == Action.Down.value:
            row += 1
        elif action == Action.Left.value:
            col -= 1
        elif action == Action.Right.value:
            col += 1

        if row < 0 or row >= len(board) or col < 0 or col >= len(board):
            return False

        if board[row, col] in [Item.Wood.value, Item.Rigid.value]:
            return False

        return True

    def find_children(self, action_priors):
        """ expands all children """

        pruned_actions = [a for a in range(6) if not self.prune(a, is_opponent=False)]
        if len(pruned_actions) == 0:
            pruned_actions = [constants.Action.Stop.value]

        for action, prob in action_priors:
            if action in pruned_actions: # added new
                for opponent_action in self.pruned_opponent_actions:
                    actions = [None, None]
                    actions[self.agent_id] = action
                    actions[1-self.agent_id] = opponent_action

                    if (actions[0], actions[1]) not in self.children.keys():
                        self.children[(actions[0], actions[1])] = self.forward(actions, prob)

    def forward(self, actions, prob):
        """ applies the actions to obtain the next game state """
        # since the forward model directly modifies the parameters, we have to provide copies
        board = copy.deepcopy(self.state[0])
        agents = _copy_agents(self.state[1])
        bombs = _copy_bombs(self.state[2])
        items = copy.deepcopy(self.state[3])
        flames = _copy_flames(self.state[4])
        board, curr_agents, curr_bombs, curr_items, curr_flames = ForwardModel.step(
            actions,
            board,
            agents,
            bombs,
            items,
            flames
        )
        return Node(self, [board, curr_agents, curr_bombs, curr_items, curr_flames], self.agent_id, prob)

    def select_greedily(self, c_puct):
        children = self.children.values()

        return max(children, key=lambda n: n.get_value(c_puct))

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent.n_visits) / (1 + self.n_visits))
        return self._Q + self._u

    def get_children(self):
        return self.children

    def update(self, reward):
        self.n_visits += 1
        self._Q += 1.0 * (reward - self._Q) / self.n_visits

    def update_recursive(self, reward):
        if self._parent:
            self._parent.update_recursive(reward)
        self.update(reward)

    def is_terminal(self):
        alive = [agent for agent in self.state[1] if agent.is_alive]
        return len(alive) != 2

    def get_total_reward(self):
        """ Returns Total reward of node (Q) """
        return self.total_reward

    def incr_reward(self, reward):
        """ Update reward of node in backpropagation step of MCTS """
        self.total_reward += reward

    def get_visit_count(self):
        """ Returns Total number of times visited this node (N) """
        return self.visit_count

    def incr_visit_count(self):
        self.visit_count += 1

    def reward(self, root_state, state_v):
        # we do not want to role out games until the end,
        # since pommerman games can last for 800 steps, therefore we need to define a value function,
        # which assigns a numeric value to state (how "desirable" is the state?)

        agents = self.state[1]
        own_agent = agents[self.agent_id]
        opponent_agent = agents[1 - self.agent_id]
        root_own_agent = root_state[1][self.agent_id]
        assert own_agent, root_own_agent
        # check if own agent is dead
        if not own_agent.is_alive:
            return -1.0
        # check if opponent has been destroyed
        elif not opponent_agent.is_alive:
            return 1.0

        return state_v


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# def _value_func(state, root_state, agent_id, actor_critic_nn, obs, device='cpu'):
#     # TODO: here you need to assign a value to a game state, for example the evaluation can
#     #   be based on the number of blasted clouds, the number of collected items the distance to the opponent, ...
#     # an example how a numerical value can be derived:
#     board = state[0]
#
#
#
#     obs_featurized = featurize_simple(obs, device=device)
#     _, score, _ = actor_critic_nn.evaluate(obs_featurized)
#     return score.item()



    # obs_opponent = obss[1 - agent_id]

    # score = 0.0  # game is not over yet, we have to think about additional evaluation criteria
    #
    # own_position = own_agent.position
    # opponent_position = opponent_agent.position
    #
    # # if agent cannot move in any direction than its locked up either by a bomb,
    # # or the opponent agent -> very bad position
    # down_cond = own_position[0] + 1 >= len(board) or \
    #     board[own_position[0] + 1][own_position[1]] not in ACCESSIBLE_TILES
    # up_cond = own_position[0] - 1 < 0 or \
    #     board[own_position[0] - 1][own_position[1]] not in ACCESSIBLE_TILES
    # right_cond = own_position[1] + 1 >= len(board) or \
    #     board[own_position[0]][own_position[1] + 1] not in ACCESSIBLE_TILES
    # left_cond = own_position[1] - 1 < 0 or \
    #     board[own_position[0]][own_position[1] - 1] not in ACCESSIBLE_TILES
    #
    # if down_cond and up_cond and right_cond and left_cond:
    #     score += -0.5
    #
    # # we want to push our agent towards the opponent
    # man_dist = manhattan_dist(own_position, opponent_position)
    # score += 0.005*(10-man_dist)  # the closer to the opponent the better
    #
    # # we want to collect items (forward model was modified to make this easier)
    # score += own_agent.picked_up_items * 0.05
    #
    # # since search depth is limited, we need to reward well placed bombs instead
    # # of only rewarding collecting items
    # for bomb in state[2]:
    #     # we only reward bombs placed next to wood - you can improve this
    #     loc = bomb.position
    #     if loc[0]-1 >= 0 and board[loc[0]-1][loc[1]] == Item.Wood.value:
    #         score += 0.02
    #     if loc[0]+1 < len(board) and board[loc[0]+1][loc[1]] == Item.Wood.value:
    #         score += 0.02
    #     if loc[1]-1 >= 0 and board[loc[0]][loc[1]-1] == Item.Wood.value:
    #         score += 0.02
    #     if loc[1]+1 < len(board) and board[loc[0]][loc[1]+1] == Item.Wood.value:
    #         score += 0.02
    # return score


def _copy_agents(agents_to_copy):
    """ copy agents of the current node """
    agents_copy = []
    for agent in agents_to_copy:
        agt = DummyAgent()
        agt.init_agent(agent.agent_id, constants.GameType.FFA)
        agt.set_start_position(agent.position)
        agt.reset(
            ammo=agent.ammo,
            is_alive=agent.is_alive,
            blast_strength=agent.blast_strength,
            can_kick=agent.can_kick
        )
        agt.picked_up_items = agent.picked_up_items
        agents_copy.append(agt)
    return agents_copy


def _copy_bombs(bombs):
    """ copy bombs of the current node """
    bombs_copy = []
    for bomb in bombs:
        bomber = characters.Bomber()
        bombs_copy.append(
            characters.Bomb(bomber, bomb.position, bomb.life, bomb.blast_strength,
                            bomb.moving_direction)
        )

    return bombs_copy


def _copy_flames(flames):
    """ copy flames of the current node """
    flames_copy = []
    for flame in flames:
        flames_copy.append(
            characters.Flame(flame.position, flame.life)
        )
    return flames_copy
