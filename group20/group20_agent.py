import pkg_resources
import os
import torch
import time
import numpy as np

from pommerman import agents, constants

from group20 import trainer
from group20.game_state import game_state_from_obs
from group20.node import Node
from group20.mcts import MCTS
from group20 import util

from group20.trainer import Trainer


class Group20Agent(agents.BaseAgent):
    """
    This is the class of your agent. During the tournament an object of this class
    will be created for every game your agents plays.
    If you exceed 500 MB of main memory used, your agent will crash.

    Args:
        ignore the arguments passed to the constructor
        in the constructor you can do initialisation that must be done before a game starts
    """
    def __init__(self, *args, **kwargs):
        super(Group20Agent, self).__init__(*args, **kwargs)

        self.model_file_name = 'model_4_finished.pt'

        self.device = torch.device("cpu")
        data_path = pkg_resources.resource_filename('group20', 'resources')
        model_file = os.path.join(data_path, self.model_file_name)

        # loading the trained neural network model
        self.trainer = trainer.Trainer()
        self.trainer.actor_critic = trainer.ActorCritic(device=self.device)
        self.trainer.actor_critic.load_state_dict(torch.load(model_file, map_location=self.device))
        self.trainer.actor_critic.to(self.device)

        self.trainer.actor_critic.eval()
        self.tree = None

        self.policy_value_fn = self.trainer.policy_value_fn

    def act(self, obs, action_space):
        """
        Every time your agent is required to send a move, this method will be called.
        You have 0.5 seconds to return a move, otherwise no move will be played.

        Parameters
        ----------
        obs: dict
            keys:
                'alive': {list:2}, board ids of agents alive
                'board': {ndarray: (11, 11)}, board representation
                'bomb_blast_strength': {ndarray: (11, 11)}, describes range of bombs
                'bomb_life': {ndarray: (11, 11)}, shows ticks until bomb explodes
                'bomb_moving_direction': {ndarray: (11, 11)}, describes moving direction if bomb has been kicked
                'flame_life': {ndarray: (11, 11)}, ticks until flame disappears
                'game_type': {int}, irrelevant for you, we only play FFA version
                'game_env': {str}, irrelevant for you, we only use v0 env
                'position': {tuple: 2}, position of the agent (row, col)
                'blast_strength': {int}, range of own bombs         --|
                'can_kick': {bool}, ability to kick bombs             | -> can be improved by collecting items
                'ammo': {int}, amount of bombs that can be placed   --|
                'teammate': {Item}, irrelevant for you
                'enemies': {list:3}, possible ids of enemies, you only have one enemy in a game!
                'step_count': {int}, if 800 steps were played then game ends in a draw (no points)

        action_space: spaces.Discrete(6)
            action_space.sample() returns a random move (int)
            6 possible actions in pommerman (integers 0-5)

        Returns
        -------
        action: int
            Stop (0): This action is a pass.
            Up (1): Move up on the board.
            Down (2): Move down on the board.
            Left (3): Move left on the board.
            Right (4): Move right on the board.
            Bomb (5): Lay a bomb.
        """
        start_time = time.time()

        if obs['step_count'] == 0:
            game_state = game_state_from_obs(obs, self.agent_id)
            root = Node(None, game_state, self.agent_id, 1.0, obs=obs)
            self.tree = MCTS(range(6), self.agent_id, root, self.policy_value_fn, device=self.device)
            # before we rollout the tree we expand the first set of children
            self.tree.expand_root(root)
        else:
            self.tree.update_root(obs, self.agent_id)

        while time.time() - start_time < 0.4:
            self.tree.do_rollout()

        bombs = util.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))

        valid_directions = [vd.value for vd in Trainer.get_valid_actions(obs['board'], obs['position'], obs['can_kick'])]
        valid_directions.append(constants.Action.Bomb.value)

        bomb_positions = [b['position'] for b in bombs] if len(bombs) > 0 else []
        bomb_blast_strengths = [b['blast_strength'] for b in bombs] if len(bombs) > 0 else []
        position = obs['position']

        _, probs = self.tree.get_move_probs()

        if position in bomb_positions:
            idx = bomb_positions.index(position)
            bomb_best_action, bomb_safe_actions = Trainer.get_best_action_bomb(
                obs['board'], position, bomb_blast_strengths[idx], valid_directions)

            if bomb_best_action is not None:
                move = bomb_best_action
            elif len(bomb_safe_actions) > 0:
                move = bomb_safe_actions[np.array(probs[np.array(bomb_safe_actions)]).argmax()]
            else:
                move = constants.Action.Stop.value
        else:
            move = self.tree.choose()

        return move
