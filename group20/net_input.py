import torch
import numpy as np
import random

from pommerman import constants

def featurize_simple(obs, device='cpu'):
    np_featurized = featurize_simple_array(obs)
    # convert to tensor and send to device
    return torch.tensor([np_featurized]).to(device)


def featurize_simple_array(obs):
    # here we encode the board observations into a structure that can
    # be fed into a convolution neural network
    board_size = len(obs['board'])

    # new big encoding:
    # - board representation:
    #   1. passage
    #   2. rigid_wall
    #   3. wooden wall
    #   4. bomb
    #   5. flames
    #   6. extra bomb powerup
    #   7. increase range powerup
    #   8. increase kick powerup
    # - position encoding
    #   9. agent position
    #   10. enemy position
    # - powerup representation
    #   11. values of ammo, blast strength and binary kick capability
    # - bomb + flame attributes (blast strength, bomb life and bomb moving direction, flame life)
    #   12. blast strength of bombs placed on the board
    #   13. bomb life
    #   14. moving direction
    #   15. flame life
    # - safe/desired cells
    #   16.
    #       powerups = 0
    #       wooden wall = 1
    #       passage = 2
    #       fog = 3
    #       enemies = 4
    #       rigid walls = 5
    #       teammate = 6
    #       bombs = 7
    #       flames = 8
    # TODO: Change board representation

    # I am not sure, but I think they already managed it for us, to switch the agents positions...

    # board representation
    board_rep = obs['board']# .astype(np.float32)

    board_passage = np.where(board_rep == constants.Item.Passage.value, 1.0, 0.0).astype(np.float32)
    board_rigid_wall = np.where(board_rep == constants.Item.Rigid.value, 1.0, 0.0).astype(np.float32)
    board_wooden_wall = np.where(board_rep == constants.Item.Wood.value, 1.0, 0.0).astype(np.float32)
    board_bomb = np.where(board_rep == constants.Item.Bomb.value, 1.0, 0.0).astype(np.float32)
    board_flames = np.where(board_rep == constants.Item.Flames.value, 1.0, 0.0).astype(np.float32)
    board_fog = np.where(board_rep == constants.Item.Fog.value, 1.0, 0.0).astype(np.float32)
    board_extra_bomb = np.where(board_rep == constants.Item.ExtraBomb.value, 1.0, 0.0).astype(np.float32)
    board_incr_range = np.where(board_rep == constants.Item.IncrRange.value, 1.0, 0.0).astype(np.float32)
    board_kick = np.where(board_rep == constants.Item.Kick.value, 1.0, 0.0).astype(np.float32)

    # to remove:
    if 1.0 in board_fog:
        print("board_fog!")

    # encode position of trainee
    position = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    position[obs['position'][0], obs['position'][1]] = 1.0

    # encode position of enemy
    enemy = obs['enemies'][0]  # we only have to deal with 1 enemy
    enemy_position = np.where(obs['board'] == enemy.value, 1.0, 0.0).astype(np.float32)

    # values of ammo, blast strength and binary kick capability
    ammo = obs['ammo']
    blast_strength = obs['blast_strength']
    kick = obs['can_kick']
    ammo_blast_kick = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    ammo_blast_kick[2, :] = ammo
    ammo_blast_kick[4:7, 2] = blast_strength
    ammo_blast_kick[4:7, 5] = blast_strength
    ammo_blast_kick[4:7, 8] = blast_strength
    ammo_blast_kick[8, :] = kick
    ammo_blast_kick = np.array(ammo_blast_kick, dtype=np.float32)

    # blast strength and leftover lives of bombs
    bomb_blast_strength = obs['bomb_blast_strength'].astype(np.float32)

    # bomb_moving_direction = obs['bomb_moving_direction'].astype(np.float32)
    flame_life = obs['flame_life'].astype(np.float32)

    bomb_life = obs['bomb_life'].astype(np.float32)
    bomb_moving_direction = obs['bomb_moving_direction'].astype(np.float32)

    desired_cells = np.where(
        (board_rep == constants.Item.ExtraBomb.value)
        | (board_rep == constants.Item.IncrRange.value)
        | (board_rep == constants.Item.Kick.value), 0, -1
    ).astype(np.float32)

    desired_cells[obs['position'][0], obs['position'][1]] = 0
    desired_cells[board_rep == constants.Item.Wood.value] = 1
    desired_cells[board_rep == constants.Item.Passage.value] = 2
    # desired_cells[board_rep == constants.Item.Fog.value] = 3
    desired_cells[board_rep == enemy.value] = 3
    desired_cells[board_rep == constants.Item.Rigid.value] = 4
    desired_cells[board_rep == constants.Item.Bomb.value] = 5
    desired_cells[board_rep == constants.Item.Flames.value] = 6

    desired_cells = desired_cells.astype(np.float32)

    # stack all the input planes
    featurized = np.stack(( # 14 + 2
        board_passage,
        board_rigid_wall,
        board_wooden_wall,
        board_bomb,
        board_flames,
        board_extra_bomb,
        board_incr_range,
        board_kick,
        position,
        enemy_position,
        ammo_blast_kick,
        bomb_blast_strength,
        bomb_life,
        bomb_moving_direction,
        flame_life,
        desired_cells
    ), axis=0)
    return featurized
