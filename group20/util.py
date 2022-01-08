import random
import copy

import pommerman
from pommerman.agents import DummyAgent, SimpleAgent
# from group20 import simple_agent_no_bombs
# from ..group20 import simple_agent_no_bombs

from group20 import simple_agent_no_bombs
from group20.game_state import game_state_from_obs

from pommerman.forward_model import ForwardModel

import numpy as np
from pommerman import constants, utility, characters


def create_training_env(training_round, percentage_done=0):
    # we need this agent just as a placeholder here, the
    # act method will not be called
    trainee = DummyAgent()
    # as an opponent the SimpleAgent is used

    if training_round == 0:
        trainee = SimpleAgent()
        opponent = SimpleAgent()
    elif training_round == 1:
        # 10 000 games
        opponent = SimpleAgent()
    elif training_round == 2:
        # 10 000 games, static agent = dummy agent

        # limit = 10
        #
        # if percentages_done < 0.1:
        #     limit = 3
        # elif percentages_done < 0.2:
        #     limit = 5
        # elif percentages_done < 0.4:
        #     limit = 7
        # elif percentages_done < 0.6:
        #     limit = 9
        # elif percentages_done < 0.8:
        #     limit = 10
        #
        # x, y = random.choice(range(limit)), random.choice(range(limit))
        #
        # bomber = characters.Bomber()
        # bomber.set_start_position((x, y))
        #
        # opponent = DummyAgent(character=bomber)

        opponent = DummyAgent()
    elif training_round == 3:
        # 20 000 games, simple agent but no bombs
        opponent = simple_agent_no_bombs.SimpleAgentNoBombs()
    elif training_round == 4:
        # 60 000 games, simple agent
        opponent = SimpleAgent()
    else:
        raise NotImplementedError()

    # we create the ids of the two agents in a randomized fashion
    ids = [0, 1]
    random.shuffle(ids)
    trainee_id = ids[0]
    opponent_id = ids[1]
    agents = [0, 0]
    agents[trainee_id] = trainee
    agents[opponent_id] = opponent
    # create the environment and specify the training agent
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.set_training_agent(trainee.agent_id)
    return env, trainee, trainee_id, opponent, opponent_id


# same methods as in node.py
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


def make_step(action, obs, trainee_id, enemy_action=0):
    """ applies the actions to obtain the next game state """
    # since the forward model directly modifies the parameters, we have to provide copies
    state = game_state_from_obs(obs, trainee_id)

    actions = [0] * 2
    actions[1-trainee_id] = enemy_action
    actions[trainee_id] = action

    board = copy.deepcopy(state[0])
    agents = _copy_agents(state[1])
    bombs = _copy_bombs(state[2])
    items = copy.deepcopy(state[3])
    flames = _copy_flames(state[4])
    board, curr_agents, curr_bombs, curr_items, curr_flames = ForwardModel.step(
        actions,
        board,
        agents,
        bombs,
        items,
        flames
    )

    return board, curr_agents


def get_next_position_with_board_copy(obs, direction, position, trainee_id):

    board, curr_agents = make_step(direction.value, obs, trainee_id)

    if direction in [constants.Action.Bomb, constants.Action.Stop]:
        new_x, new_y = position
    else:
        new_x, new_y = utility.get_next_position(position, direction)

        if not utility.position_on_board(board, (new_x, new_y)):
            return board, None

    if board[new_x, new_y] in [constants.Item.Rigid.value,
                               constants.Item.Wood.value,
                               constants.Item.Flames.value]:
        return board, None

    return board, (new_x, new_y)


    #
    #     old_x, old_y = position
    #
    #     board[new_x, new_y] = board[old_x, old_y]
    #     board[old_x, old_y] = constants.Item.Passage.value
    #
    # if 1 in bomb_lifes:
    #     for i, (bomb_x, bomb_y) in enumerate(bomb_positions):
    #         if bomb_lifes[i] == 1:
    #             # not completly correct, but good enough
    #             board[bomb_x-blast_strength+1:bomb_x+blast_strength, bomb_y] = constants.Item.Flames.value
    #             board[bomb_x, bomb_y-blast_strength+1:bomb_y+blast_strength] = constants.Item.Flames.value
    #
    # if 1 in obs_flame_lifes:
    #     board[obs_flame_lifes == 1] = constants.Item.Passage.value
    #
    # return board, (new_x, new_y)


def find_bomb_coverings(board, position):
    x, y = position
    row = board[:, y]
    column = board[x, :]

    def manhatten_distance(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        return abs(x2 - x1) + abs(y2 - y1)

    row_distances, column_distances, row_bombs, column_bombs = [], [], [], []

    row_xs, column_ys = [], []

    row_xs_unchecked = np.where(row == constants.Item.Bomb.value)[0]
    column_ys_unchecked = np.where(column == constants.Item.Bomb.value)[0]

    if len(row_xs_unchecked) == 0 and len(column_ys_unchecked) == 0:
        return [np.inf, -np.inf]

    if len(row_xs_unchecked) > 0:
        # filter for bombs without rigid between position and bombs position
        for x_index in row_xs_unchecked:
            elements_between = board[min(x,x_index):max(x,x_index), y]

            if constants.Item.Rigid.value not in elements_between:
                row_xs.append(x_index)

        if len(row_xs) > 0:
            row_xs = np.array(row_xs)
            row_ys = np.full_like(row_xs, y)
            try:
                if len(row_xs) > 1:
                    row_bombs = np.stack([row_xs, row_ys], axis=1)
                    row_distances = [manhatten_distance(bomb, position) for bomb in row_bombs]
                else:
                    row_bombs = [np.concatenate([row_xs, row_ys], axis=0)]
                    row_distances = [manhatten_distance(row_bombs[0], position)]
            except Exception as e:
                print("NOW FAIL", row_bombs, position)
                print(e)

    if len(column_ys_unchecked) > 0:
        # filter for bombs without rigid between position and bombs position
        for y_index in column_ys_unchecked:
            elements_between = board[x, min(y, y_index):max(y, y_index)]

            if constants.Item.Rigid.value not in elements_between:
                column_ys.append(y_index)

        if len(column_ys) > 0:
            column_ys = np.array(column_ys)
            column_xs = np.full_like(column_ys, x)
            try:
                if len(column_ys) > 1:
                    column_bombs = np.stack([column_xs, column_ys], axis=1)
                    column_distances = [manhatten_distance(bomb, position) for bomb in column_bombs]
                else:
                    column_bombs = [np.concatenate([column_xs, column_ys], axis=0)]
                    column_distances = [manhatten_distance(column_bombs[0], position)] # [0]
            except Exception as e:
                print("NOW FAIL", column_bombs, position)
                print(e)

    if len(row_xs) == 0 and len(column_ys) == 0:
        return [np.inf, -np.inf]

    if len(row_distances) == 0 or len(column_distances) == 0:
        distances = row_distances if len(row_distances) > len(column_distances) else column_distances
    else:
        distances = np.concatenate([row_distances, column_distances])

    distances.sort()

    if len(distances) == 1:
        distances = [distances[0], np.inf]

    return distances


def find_min_bomb_covering(board, position):
    return find_bomb_coverings(board, position)[0]


def find_max_bomb_covering(board, position):
    return find_bomb_coverings(board, position)[-1]


def find_max_min_bomb_covering(board, position):
    c = find_bomb_coverings(board, position)
    return c[-1], c[0]


"""
    When does it return 0 and when -np.inf ???? 
    Calculates the minimum amount of evade steps needed. 
    
    History: History of visited positions. 
    
    n should be as small as possible
"""


def min_evade_step(board, p, history, blast_strength, invalid_values):
    u, l = find_max_min_bomb_covering(board, p) # if not bomb_on_position else (0, 0)
    if u == -np.inf:
        # no bomb covering p
        return 0
    elif len(history) >= u:
        if len(history) > u + blast_strength:
            # we escape the bomb
            return 0
        else:
            # even the bomb "most far away" would have exploded upon arrival
            return np.inf
    elif len(history) >= l:
        if len(history) > l + blast_strength:
            # we escape the bomb
            return 0
        else:
            # we cannot even escape the nearest bomb
            return np.inf

    # if the amount of steps taken (len(history)) is smaller than the amount of steps needed to escape the nearest bomb
    num = np.inf
    for a in [constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]:

        q = utility.get_next_position(p, a)

        if q not in history and utility.is_valid_direction(board, p, a, invalid_values): # utility.position_on_board(board, q):
            history.append(q)

            num = min(num, 1 + min_evade_step(board, q, history, blast_strength, invalid_values))

    return num


def get_jitter_details(xposition, yposition):
    jitter_amount = 0
    jitter_actions = []

    static_cond1 = len(set(xposition[-15:])) == 1
    static_cond2 = len(set(yposition[-15:])) == 1

    x_cond_odd = len(set(xposition[-10::2])) == 1
    x_cond_even = len(set(xposition[-11::2])) == 1
    x_cond_uneq = not (len(set(xposition[-11::2]) - set(xposition[-10::2])) == 0)
    x_cond_long = len(set(xposition[-35:])) == 2
    x_y_cond_long = len(set(xposition[-35:])) == 1

    y_cond_odd = len(set(yposition[-10::2])) == 1
    y_cond_even = len(set(yposition[-11::2])) == 1
    y_cond_uneq = not (len(set(yposition[-11::2]) - set(yposition[-10::2])) == 0)
    y_cond_long = len(set(yposition[-35:])) == 2
    y_x_cond_long = len(set(yposition[-35:])) == 1

    # adding bomb action and stoping action just to go sure, we can again bomb after the 2/3 steps
    if static_cond1 and static_cond2:
        # stuck - doesnt move
        jitter_amount = 3  # take next 3 steps from expert policy
        jitter_actions = [constants.Action.Stop.value, constants.Action.Bomb.value]
    elif (x_cond_odd and x_cond_even and x_cond_uneq) or (x_cond_long and x_y_cond_long):
        # stuck between actions up and down (x-axis and y-axis are switched)
        jitter_amount = 2  # take next 2 steps from expert policy
        jitter_actions = [constants.Action.Up.value, constants.Action.Down.value,
                          constants.Action.Bomb.value, constants.Action.Stop.value]
    elif (y_cond_odd and y_cond_even and y_cond_uneq) or (y_cond_long and y_x_cond_long):
        # stuck between actions left and right (x-axis and y-axis are switched)
        jitter_amount = 2  # take next 2 steps from expert policy
        jitter_actions = [constants.Action.Right.value,
                          constants.Action.Left.value,
                          constants.Action.Bomb.value,
                          constants.Action.Stop.value]

    return jitter_amount, jitter_actions


def convert_bombs(bomb_map, bomb_life):
    '''Flatten outs the bomb array'''
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({
            'position': (r, c),
            'blast_strength': int(bomb_map[(r, c)]),
            'bomb_life': int(bomb_life[(r, c)])
        })
    return ret