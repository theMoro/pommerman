from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, Categorical
import time
from torch.autograd import Variable

from pommerman import utility, constants

from group20.replay_memory import ReplayMemory, Transition
from group20 import util
from group20.game_state import game_state_from_obs
from group20.node import Node
from group20.mcts import MCTS
from group20.net_input import featurize_simple, featurize_simple_array

import numpy as np
import random


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ActorCritic(nn.Module):
    def __init__(self, board_size=11, num_boards=16, num_actions=6, lr=1e-3, device='cpu'):
        super(ActorCritic, self).__init__()

        self.device = device

        self.board_size = board_size
        self.num_boards = num_boards
        self.num_actions = num_actions

        # common layers
        self.conv1 = nn.Conv2d(in_channels=num_boards, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.b_norm_1 = nn.BatchNorm2d(32)
        # relu
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.b_norm_2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.b_norm_3 = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()

        # action policy layers
        self.a_fc1 = nn.Linear(648, 128)
        self.a_fc2 = nn.Linear(128, num_actions)

        # state value layers
        self.c_fc1 = nn.Linear(648, 128)
        self.c_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.b_norm_1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.b_norm_2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.b_norm_3(x))

        x = self.flatten(x)

        x_act = F.relu(self.a_fc1(x))
        x_act = F.log_softmax(self.a_fc2(x_act), dim=-1)

        x_val = F.relu(self.c_fc1(x))
        x_val = torch.tanh(self.c_fc2(x_val))

        return x_act, x_val


class Trainer():
    def __init__(self,
                 board_size=11,
                 num_boards=16,
                 num_actions=6,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=16,
                 device='cpu'
                 ):

        self.lr = lr
        self.lr_multiplier = 1.0
        self.kl_targ = 0.02

        self.gamma = gamma
        self.k_epochs = 5

        self.batch_size = batch_size

        self.actor_critic = ActorCritic(board_size, num_boards, num_actions,
                                        lr=lr, device=device)

        self.tree = None

        self.MseLoss = nn.MSELoss()

        self.l2_const = 1e-4

        self.optimizer = optim.Adam(params=self.actor_critic.parameters(), lr=lr, weight_decay=self.l2_const)

    @staticmethod
    def get_neighbour_positions(pos):
        x, y = pos
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    @staticmethod
    def is_walkable_tile(board, pos, extra_item=None):
        items = [constants.Item.Passage.value, constants.Item.ExtraBomb.value,
                 constants.Item.IncrRange.value, constants.Item.Kick.value]

        if extra_item is not None:
            items.append(extra_item.value)

        return board[pos] in items

    @staticmethod
    def in_dangerous_position(board, pos, bombs, bomb_lifes, blast_strength):
        for i, bomb in enumerate(bombs):
            if 0 < bomb_lifes[i] <= 2:
                if bomb[0] == pos[0]:
                    b = bomb[1]
                    p = pos[1]
                    walkable = True

                    for middle in range(min(b, p) + 1, max(b, p)):
                        if not Trainer.is_walkable_tile(board, (pos[0], middle), extra_item=constants.Item.Wood):
                            walkable = False

                    if walkable and abs(p - b) <= blast_strength:
                        return True

                if bomb[1] == pos[1]:
                    b = bomb[0]
                    p = pos[0]
                    walkable = True

                    for middle in range(min(b, p) + 1, max(b, p)):
                        if not Trainer.is_walkable_tile(board, (middle, pos[1]), extra_item=constants.Item.Wood):
                            walkable = False

                    if walkable and abs(p - b) <= blast_strength:
                        return True

        return False

    @staticmethod
    def get_valid_actions(board, position, can_kick, actions=None):

        if actions is None:
            actions = [
                constants.Action.Stop, constants.Action.Left,
                constants.Action.Right, constants.Action.Up, constants.Action.Down
            ]

        invalid_values = [item.value for item in [constants.Item.Rigid, constants.Item.Wood, constants.Item.Flames]] # Flames

        if not can_kick:
            invalid_values.append(constants.Item.Bomb.value)

        valid_directions = [a for a in actions
                            if utility.is_valid_direction(board, position, a, invalid_values=invalid_values)]

        return valid_directions

    """
        Action Filter function
    """

    @staticmethod
    def get_safe_actions(board, obs, trainee_id,
                         position, blast_strength, bombs, can_kick, obs_flame_life, actions=None):

        bomb_positions = [b['position'] for b in bombs] if len(bombs) > 0 else []
        bomb_lifes = [b['bomb_life'] for b in bombs] if len(bombs) > 0 else []

        if actions is None:
            actions = [
                constants.Action.Stop, constants.Action.Left,
                constants.Action.Right, constants.Action.Up, constants.Action.Down
            ]

        invalid_values = [item.value for item in [constants.Item.Rigid, constants.Item.Wood, constants.Item.Flames]]

        if not can_kick:
            invalid_values.append(constants.Item.Bomb.value)

        valid_directions = Trainer.get_valid_actions(board, position, can_kick, actions)

        safe_actions = []
        more_dangerous_actions = []

        if len(bomb_positions) > 0:
            for a in actions:
                if a in valid_directions:
                    # board_, q = util.get_next_position_with_board_copy(board, position, bomb_positions,
                    #                                                    bomb_lifes, obs_flame_life,
                    #                                                    blast_strength, a)

                    board_, q = util.get_next_position_with_board_copy(obs, a, position, trainee_id)
                    if q is not None and not (q in bomb_positions and not can_kick):
                        if a in [constants.Action.Stop, constants.Action.Bomb]:
                            H = [None, q]
                        else:
                            H = [q]

                        n = util.min_evade_step(board_, q, H, blast_strength, invalid_values)
                        m = util.find_min_bomb_covering(board_, position)

                        if m > n:
                            # if i am sitting on a bomb any action (especially away from it) is considered safe
                            if position not in bomb_positions and \
                                    Trainer.in_dangerous_position(board_, q, bomb_positions, bomb_lifes,
                                                                  blast_strength):
                                more_dangerous_actions.append(a.value)
                            else:
                                safe_actions.append(a.value)

        all_possibly_safe_actions = safe_actions.copy()

        for dang in more_dangerous_actions:
            all_possibly_safe_actions.append(dang)

        if len(safe_actions) == 0:
            if len(all_possibly_safe_actions) == 0:
                all_possibly_safe_actions = [v.value for v in valid_directions]

            safe_actions = all_possibly_safe_actions

        if constants.Action.Stop.value in all_possibly_safe_actions:
            all_possibly_safe_actions.append(constants.Action.Bomb.value)

        return safe_actions, all_possibly_safe_actions

    def get_penalty(self, selected_action, valid_directions, board, position, enemy, bombs,
                    blast_strength, ammo, can_kick, recently_visited_positions):

        # valid directions are with bomb value !!!

        def get_positions_in_blast_zone(pos):
            positions = [(pos[0], pos[1])]

            for blast in range(blast_strength):
                if pos[0] + blast <= 10:
                    positions.append((pos[0] + blast, pos[1]))

                if pos[1] + blast <= 10:
                    positions.append((pos[0], pos[1] + blast))

                if pos[0] - blast >= 0:
                    positions.append((pos[0] - blast, pos[1]))

                if pos[1] - blast >= 0:
                    positions.append((pos[0], pos[1] - blast))

            return positions

        # invalid direction
        if selected_action not in valid_directions:
            return -0.03

        # bomb and no ammo
        if selected_action == constants.Action.Bomb.value and ammo <= 0:
            return -0.03

        # picking up things
        if selected_action != constants.Action.Bomb.value:
            new_x, new_y = utility.get_next_position(position, constants.Action(selected_action.item()))

            item = board[new_x, new_y]
            if item in [constants.Item.Kick.value, constants.Item.IncrRange.value, constants.Item.ExtraBomb.value]:
                return 0.02

            # kicking bomb
            if item == constants.Item.Bomb.value:
                if can_kick:
                    return 0.02
                else:
                    return -0.02

            # new position
            if (new_x, new_y) not in recently_visited_positions:
                return 0.001

        # (trying to) bomb walls or enemy
        if selected_action == constants.Action.Bomb.value:
            positions_in_blast_zone = get_positions_in_blast_zone(position)
            board_items = [board[pos[0], pos[1]] for pos in positions_in_blast_zone]

            if constants.Item.Wood.value in board_items:
                return 0.02

            # we know that our agent is in the zone, so the question is if our enemy is, too
            if 10 in board_items and 11 in board_items:
                return 0.05

        return 0

    # state = obs_featurized
    def select_action_mcts(self, state, obs, agent_id,
                           jitter_actions, recently_visited_positions, new_game, device):

        jitter_action = None
        jitter_success = False

        if new_game:
            self.tree = None

        board = obs['board']
        position = obs['position']
        blast_strength = obs['blast_strength']
        enemies = [constants.Item(e) for e in obs['enemies']]
        bombs = util.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))
        ammo = int(obs['ammo'])
        can_kick = bool(obs['can_kick'])
        flame_life = obs['flame_life']

        valid_directions = [vd.value for vd in self.get_valid_actions(board, position, can_kick)]
        valid_directions.append(constants.Action.Bomb.value)

        # jitter
        if len(jitter_actions) > 0:
            vd_directions = valid_directions.copy()
            safe_actions, all_possibly_safe_actions = self.get_safe_actions(board, obs, agent_id, position,
                                                                            blast_strength, bombs, can_kick, flame_life)
            original_safe_actions = safe_actions.copy()
            for a in jitter_actions:
                if a in safe_actions:
                    safe_actions.remove(a)

            if len(safe_actions) == 0:
                for a in jitter_actions:
                    if a in vd_directions:
                        vd_directions.remove(a)

                if len(vd_directions) > 0:
                    jitter_action = torch.tensor(np.random.choice(vd_directions, size=1),
                                                 device=device, dtype=torch.long)
                else:
                    jitter_action = torch.tensor(np.random.choice(original_safe_actions, size=1),
                                                 device=device, dtype=torch.long)
            else:
                jitter_action = torch.tensor(np.random.choice(safe_actions, size=1),
                                             device=device, dtype=torch.long)

        # start_time = time.time() # to delete

        # tree part
        # look at obs
        if self.tree is None:  # OUT-COMMENT THIS !!!
            game_state = game_state_from_obs(obs, agent_id)
            root = Node(None, game_state, agent_id, 1.0, obs=obs)
            self.tree = MCTS(range(6), agent_id, root, self.policy_value_fn, device=device)
            # before we rollout the tree we expand the first set of children
            self.tree.expand_root(root)

        self.tree.set_root_obs_state(obs)  # try to not use this

        for _ in range(20):  # tune parameter
            self.tree.do_rollout()

        # while time.time() - start_time < 0.45:
        #     self.tree.do_rollout()

        # get tree loss
        move_probs = np.zeros(6)
        acts, probs = self.tree.get_move_probs()
        move_probs[np.array(acts)] = probs

        if jitter_action is not None:
            move = jitter_action
            jitter_success = True
        else:
            move = np.random.choice(acts,
                                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            move = torch.tensor([move], dtype=torch.float, device=device)

        penalty = self.get_penalty(move, valid_directions, board, position, enemies, bombs,
                                   blast_strength, ammo, can_kick, recently_visited_positions)

        return move, move_probs, penalty, jitter_success  # , penalty, False

    def policy_value_fn(self, obs, agent_id, device='cpu'):
        board = obs['board']
        position = obs['position']
        blast_strength = obs['blast_strength']
        enemies = [constants.Item(e) for e in obs['enemies']]
        bombs = util.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))
        ammo = int(obs['ammo'])
        can_kick = bool(obs['can_kick'])
        flame_life = obs['flame_life']

        valid_directions = [vd.value for vd in self.get_valid_actions(board, position, can_kick)]
        valid_directions.append(constants.Action.Bomb.value)

        safe_actions, all_possibly_safe_actions = self.get_safe_actions(board, obs, agent_id, position, blast_strength,
                                                                        bombs, can_kick, flame_life)

        state = featurize_simple(obs, device=device)
        log_act_probs, value = self.actor_critic(state)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy()[0])

        act_probs = zip(safe_actions, act_probs[safe_actions])
        # act_probs are action_nr + probability pairs
        return act_probs, value.item()

    def calculate_returns(self, rewards, terminals, discount_factor, normalize=True, device='cpu'):
        returns = []
        R = 0

        for r, terminal in zip(reversed(rewards), reversed(terminals)):
            if terminal:
                R = 0

            R = r + (R * discount_factor)
            returns.insert(0, R)

        returns = torch.tensor(returns, device=device)

        if normalize:
            returns = (returns - returns.mean()) / max(returns.std(), 1e-5)

        return returns

    def calculate_advantages(self, rewards, values, terminals, gamma, gae_lambda, normalize=True, device='cpu'):
        advantages = []
        advantage = 0
        next_value = 0

        masks = torch.FloatTensor(~terminals.detach().cpu().numpy()).to(device)

        for r, v, mask in zip(reversed(rewards), reversed(values), reversed(masks)):
            td_error = r + next_value * gamma * mask - v
            advantage = td_error + advantage * gamma * gae_lambda * mask
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages, device=device)

        if normalize:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-5)

        return advantages

    def calculate_gae(self, rewards, penalties, values, terminals, gamma, gae_lambda, normalize=True, device='cpu'):
        returns = []
        gae = 0
        next_value = 0

        masks = torch.FloatTensor(~terminals.detach().cpu().numpy()).to(device)

        for r, p, v, mask in zip(reversed(rewards), reversed(penalties), reversed(values), reversed(masks)):
            td_error = r + p + next_value * gamma * mask - v  # r + my_reward_fct + next_value * gamma ...
            gae = td_error + gae * gamma * gae_lambda * mask
            next_value = v
            returns.insert(0, gae + v)

        returns = torch.tensor(returns, device=device)

        if normalize:
            returns = (returns - returns.mean()) / max(returns.std(), 1e-5)

        return returns

    def update_step(self, states, mcts_action_probs, rewards, penalties, terminals, device='cpu'):
        self.optimizer.zero_grad()

        old_log_probs, old_v = self.actor_critic(states)
        returns = self.calculate_gae(rewards, penalties, old_v, terminals, 0.99, 0.95, device=device)

        # maybe del
        returns = Variable(returns)

        value_loss = F.mse_loss(old_v.view(-1), returns)  # rewards
        policy_loss = -torch.mean(torch.sum(mcts_action_probs * old_log_probs, 1))  # old_probs
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()

        set_learning_rate(self.optimizer, self.lr*self.lr_multiplier)

        loss.backward()

        # clip gradients
        for param in self.actor_critic.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return policy_loss, value_loss, loss

    def update_mcts(self, replay_memory, horizon, device='cpu'):
        # 'state', 'action', 'action_probs', 'reward', 'terminal', 'penalty'
        transitions = replay_memory.sample(horizon)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        mcts_action_probs = torch.cat(batch.action_probs)
        rewards = torch.cat(batch.reward)
        terminals = torch.cat(batch.terminal)
        penalties = torch.cat(batch.penalty)

        # maybe not necessary:
        states = Variable(states)
        mcts_action_probs = Variable(mcts_action_probs)

        old_log_probs, old_v = self.actor_critic(states)
        old_probs = torch.exp(old_log_probs)

        kl = 0
        policy_loss, value_loss, loss = None, None, None

        for i in range(self.k_epochs):
            policy_loss, value_loss, loss = \
                self.update_step(states, mcts_action_probs, rewards, penalties, terminals, device=device)

            new_log_probs, new_v = self.actor_critic(states)
            # new_probs = torch.exp(old_log_probs)

            kl = torch.mean(torch.sum(old_probs * (old_log_probs - new_log_probs), dim=1))

            if kl > self.kl_targ * 4:
                break

        # adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return policy_loss.item(), value_loss.item(), loss.item()
