"""
This file shows a basic setup how a reinforcement learning agent
can be trained using DQN. If you are new to DQN, the code will probably be
not sufficient for you to understand the whole algorithm. Check out the
'Literature to get you started' section if you want to have a look at
additional resources.
Note that this basic implementation will not give a well performing agent
after training, but you should at least observe a small increase of reward.
"""
import numpy as np
import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F
import random
import os
import torch.optim as optim

from group20 import group20_agent
from pommerman import agents
import pommerman
from group20 import util
from group20.net_input import featurize_simple, featurize_simple_array
from group20.trainer import Trainer, ActorCritic
from group20.replay_memory import ReplayMemory, Transition
from pommerman import constants

from tqdm import tqdm  # DEINSTALL tqdm # REMOVE !!!!


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


''' 
    by RL-part: change from greedy selection to UCB1/ UCT
'''


def train(device_name="cuda:0", model_folder="group20/resources", model_file="imitation_model.pt",
          load_imitation_model=False, k_epochs=4, lr_actor_imitation=1e-3, lr=1e-3,
          render=False, batch_size=16, gamma=0.99, print_stats=100, save_model=100,
          min_horizon=1024):
    device = device_name
    if device_name == 'cuda:0':
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

    tqdm.write("Running on device: {}".format(device))

    model_path = os.path.join(model_folder, model_file)

    # create the environment
    env, trainee, trainee_id, opponent, opponent_id = util.create_training_env(
        training_round=0)  # training round not always 1 at the beginning
    # resetting the environment returns observations for both agents
    current_state = env.reset()
    obs_trainee = current_state[trainee_id]
    obs_opponent = current_state[opponent_id]
    # featurize observations, such that they can be fed to a neural network
    obs_trainee_featurized = featurize_simple(obs_trainee, device)
    obs_size = obs_trainee_featurized.size()

    num_boards = obs_size[1]
    board_size = obs_size[2]
    num_actions = env.action_space.n

    training_agent = Trainer(board_size, num_boards, num_actions, lr, gamma, batch_size,
                             device)

    if load_imitation_model:
        training_agent.actor_critic.load_state_dict(torch.load(model_path, map_location=device))

    training_agent.actor_critic.to(device)

    replay_memory = ReplayMemory(10000)

    # region --- TRAIN ACTOR NETWORK (Imitation network) ---
    if not load_imitation_model:
        imitation_training_games = 50000 - (50000 % batch_size)
        training_round = 0
        loss_function = F.cross_entropy
        training_agent.optimizer = torch.optim.Adam(
            params=training_agent.actor_critic.parameters(), lr=lr_actor_imitation)
        training_agent.actor_critic.train()

        tqdm.write('Training actor network')

        errors = [0]
        game_nr = 0

        with tqdm(total=imitation_training_games) as pbar:
            while game_nr <= imitation_training_games:

                action_list = []
                featurized_input = []
                print_game_nr = game_nr

                for u in range(batch_size + 1):
                    actions = [0] * 2
                    actions[trainee_id] = trainee.act(obs_trainee, env.action_space.n)
                    actions[opponent_id] = opponent.act(obs_opponent, env.action_space.n)

                    if u > 0:
                        obs_trainee_featurized = featurize_simple_array(obs_trainee)
                        obs_opponent_featurized = featurize_simple_array(obs_opponent)

                        featurized_input.append(obs_trainee_featurized)
                        action_list.append(actions[trainee_id])

                        featurized_input.append(obs_opponent_featurized)
                        action_list.append(actions[opponent_id])

                    current_state, _, terminal, _ = env.step(actions)

                    if terminal:
                        if render:
                            env.render()
                        env.close()

                        env, trainee, trainee_id, opponent, opponent_id = util.create_training_env(
                            training_round=training_round)

                        current_state = env.reset()
                        obs_trainee = current_state[trainee_id]
                        obs_opponent = current_state[opponent_id]

                        game_nr += 1
                        pbar.update(1)

                        if game_nr % save_model == 0:
                            torch.save(training_agent.actor_critic.state_dict(), model_path)

                        if game_nr % print_stats == 0 and game_nr > 0:
                            di = ((print_game_nr + 1) % print_stats)
                            di = print_stats if di == 0 else di
                            error_to_print = errors[-1] / di
                            tqdm.write('%d --- another %d games finished with training loss %g' %
                                       (game_nr, di, error_to_print))
                            errors.append(0)

                        if game_nr == 5000:
                            set_learning_rate(training_agent.optimizer, 3e-4)
                        if game_nr == 10000:
                            set_learning_rate(training_agent.optimizer, 1e-4)
                        if game_nr == 15000:
                            set_learning_rate(training_agent.optimizer, 3e-5)
                        if game_nr == 25000:
                            set_learning_rate(training_agent.optimizer, 1e-5)

                    else:
                        obs_trainee = current_state[trainee_id]
                        obs_opponent = current_state[opponent_id]

                featurized_input_tensor = torch.tensor(featurized_input).to(device)
                action_tensor = torch.tensor(action_list).to(device)

                # maybe shuffle them
                random_indices = torch.randperm(len(featurized_input_tensor))
                featurized_input_tensor = featurized_input_tensor[random_indices]
                action_tensor = action_tensor[random_indices]

                training_agent.optimizer.zero_grad()

                preds, _ = training_agent.actor_critic(featurized_input_tensor)  #
                error = loss_function(preds.squeeze(dim=1), action_tensor)
                errors[-1] += error.item()
                error.backward()

                training_agent.optimizer.step()

        torch.save(training_agent.actor_critic.state_dict(), model_path)
        tqdm.write("--- just saved model ---")
        tqdm.write('Imitation training finished')
        tqdm.write('Losses: ' + ', '.join([f'{err / print_stats:.2f}' for err in errors]))
    # endregion

    # region --- TRAIN CRITIC NETWORK ---
    # episodes_per_training_round = [10000, 10000, 10000, 60000]
    episodes_per_training_round = [1000, 2000, 2000, 6000] # removed 0s # 1000
    # print_stats = 10
    # episodes_per_training_round = [amount - (amount % batch_size) for amount in episodes_per_training_round]

    training_round = 1

    # create new randomized environment
    env, trainee, trainee_id, opponent, opponent_id = util.create_training_env(
        training_round=training_round)

    # resetting the environment returns observations for both agents
    current_state = env.reset()
    # featurize observations, such that they can be fed to a neural network
    obs_trainee = current_state[trainee_id]
    obs_trainee_featurized = featurize_simple(obs_trainee, device)

    obs_opponent = current_state[opponent_id]

    training_agent.optimizer = optim.Adam(
        params=training_agent.actor_critic.parameters(), lr=lr, weight_decay=training_agent.l2_const)
    training_agent.actor_critic.train()

    model_files = ['model_1.pt', 'model_2.pt', 'model_3.pt', 'model_4.pt']
    action_strings = ['Stop', 'Up', 'Down', 'Left', 'Right', 'Bomb']

    for z in range(0, len(episodes_per_training_round)):
        episodes = episodes_per_training_round[z]
        tqdm.write('Critic network training round %d (%g games)' % (training_round, episodes))

        model_path = os.path.join(model_folder, model_files[z])

        episode_count = 0

        jitter_amount = 0
        jitter_actions = []

        xposition, yposition = [], []
        recently_visited_positions = []
        opponent_recently_visited_positions = []

        replay_memory.memory = []
        replay_memory.position = 0

        reward_count = 0
        wins = 0

        # losses
        action_losses, critic_losses = 0, 0
        losses = 0

        nr_iterations = 0
        nr_updates = 1

        last_update_iteration_nr = 0

        new_game = True
        actions_taken = [0] * 6

        # training loops
        with tqdm(total=episodes) as pbar:
            while episode_count < episodes:
                if render:
                    env.render()

                # jitter correction
                if jitter_amount == 0:
                    jitter_correction_eps = 0.1
                    jitter_correction_sample = random.random()

                    jitter_obs = current_state[trainee_id]
                    jitter_x, jitter_y = jitter_obs['position']

                    xposition.append(jitter_x)
                    yposition.append(jitter_y)

                    if jitter_correction_sample <= jitter_correction_eps:
                        # Note: the x-axis and y-axis are "switched"
                        jitter_amount, jitter_actions = util.get_jitter_details(xposition,
                                                                                yposition)

                    xposition = xposition[-35:]
                    yposition = yposition[-35:]

                # action, logprob, penalty, jitter_success
                action, action_probs, penalty, jitter_success = \
                    training_agent.select_action_mcts(obs_trainee_featurized, obs_trainee, trainee_id,
                                                      jitter_actions,
                                                      recently_visited_positions, new_game,
                                                      device)

                new_game = False
                actions_taken[int(action.item())] += 1

                if jitter_amount > 0 and jitter_success:
                    if jitter_amount == 1:
                        jitter_actions = []
                        xposition = []
                        yposition = []

                    jitter_amount -= 1

                # taking a step in the environment by providing actions of both agents
                actions = [0] * 2
                actions[trainee_id] = action.item()
                # getting action of opponent
                actions[opponent_id] = opponent.act(obs_opponent, env.action_space.n)

                current_state, both_rewards, terminal, info = env.step(actions)
                obs_trainee_featurized_next = featurize_simple(current_state[trainee_id], device)

                # set new root of tree
                if training_agent.tree is not None:
                    last_selected_actions = actions.copy()
                    training_agent.tree.update_root(last_selected_actions, current_state[trainee_id], trainee_id)

                # preparing transition (s, a, r, s', terminal) to be stored in replay buffer
                penalty = torch.tensor([penalty], device=device)
                reward = float(both_rewards[trainee_id])
                reward = torch.tensor([reward], device=device)
                terminal = torch.tensor([terminal], device=device, dtype=torch.bool)
                action_probs = torch.tensor([action_probs], device=device)

                replay_memory.push(obs_trainee_featurized, action, action_probs, reward, terminal, penalty)

                if terminal:
                    # optimize model
                    if (nr_iterations - last_update_iteration_nr) >= min_horizon:
                        diff = nr_iterations - last_update_iteration_nr

                        al, cl, ls = \
                            training_agent.update_mcts(replay_memory, diff, device=device)

                        # al, cl, ls = \
                        #     training_agent.update(replay_memory, diff,
                        #                      training_round, device=device)

                        replay_memory.memory = []
                        replay_memory.position = 0

                        action_losses += (abs(al) / diff)
                        critic_losses += (abs(cl) / diff)
                        losses += (abs(ls) / diff)

                        nr_updates += 1
                        last_update_iteration_nr = nr_iterations

                    episode_count += 1
                    pbar.update(1)

                    reward_count += reward.item()
                    if reward.item() == 1:
                        wins += 1

                    if render:
                        env.render()
                    env.close()

                    recently_visited_positions = []
                    opponent_recently_visited_positions = []
                    jitter_amount = 0
                    jitter_actions = []

                    # create new randomized environment
                    env, trainee, trainee_id, opponent, opponent_id = util.create_training_env(
                        training_round=training_round, percentage_done=(episode_count / episodes))

                    current_state = env.reset()

                    new_game = True

                    obs_trainee = current_state[trainee_id]
                    obs_trainee_featurized = featurize_simple(obs_trainee, device)

                    obs_opponent = current_state[opponent_id]

                    if episode_count % save_model == 0 and episode_count > 0:
                        torch.save(training_agent.actor_critic.state_dict(), model_path)

                    if (episode_count % print_stats == 0 and episode_count > 0) or episode_count - 1 == episodes:
                        action_percentages_str = ', '.join([f"{action_strings[i]}: "
                                                            f"{((actions_taken[i] / sum(actions_taken)) * 100):.2f}% "
                                                            f"({actions_taken[i]})" for i in
                                                            range(len(actions_taken))])
                        actions_taken = [0] * 6
                        tqdm.write("Episode: {}, Reward: {}, Wins: {}, Nr of iterations: {}, "
                                   "Action percentages: {}".format(
                            episode_count, reward_count, wins,
                            nr_iterations, action_percentages_str))

                        action_losses *= min_horizon
                        critic_losses *= min_horizon
                        losses *= min_horizon

                        if nr_updates > 1:
                            nr_updates -= 1

                        action_losses /= nr_updates
                        critic_losses /= nr_updates
                        losses /= nr_updates

                        tqdm.write(f'loss: {losses:.5f}, action_loss: {action_losses:.5f}, '
                                   f'critic_loss: {critic_losses:.5f}')

                        nr_updates = 1

                        action_losses, critic_losses = 0, 0
                        losses = 0

                        reward_count = 0
                        wins = 0
                else:
                    obs_trainee = current_state[trainee_id]
                    obs_trainee_featurized = obs_trainee_featurized_next

                    obs_opponent = current_state[opponent_id]

                    recently_visited_positions.append(obs_trainee['position'])
                    recently_visited_positions = recently_visited_positions[-121:]

                    opponent_recently_visited_positions.append(obs_opponent['position'])
                    opponent_recently_visited_positions = opponent_recently_visited_positions[-121:]

                nr_iterations += 1
        training_round += 1

        torch.save(training_agent.actor_critic.state_dict(), model_path)
        tqdm.write("--- just saved models ---")
    # endregion


if __name__ == "__main__":
    device = 'cuda:0'
    model = os.path.join("group20", "resources")
    # print("change name") # render
    train(device_name=device, model_folder=model, load_imitation_model=True,
          render=False, model_file='imitation_model.pt')  # To change
