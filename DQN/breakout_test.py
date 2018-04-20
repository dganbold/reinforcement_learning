
import gym
from DeepQLearner import *
from PlotWrapper import *
import atari_wrappers
#
#
def make_env(env_name):
    env = atari_wrappers.make_atari(env_name)
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
    return env
#
#
if __name__ == "__main__":
    # ----------------------------------------
    params = atari_wrappers.HYPERPARAMS['breakout-small']
    #BreakoutNoFrameskip-v4
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']
    epsilon_floor = params['epsilon_final']
    exploration_slope = (params['epsilon_final'] - params['epsilon_start'])/params['epsilon_frames']
    # ----------------------------------------
    # Define parameters for Q-learning
    gamma = params['gamma'] #0.98
    epoch = 20
    # Define parameters for Q-network
    hidden_neurons = 256
    update_target = params['target_net_sync']
    max_memory = params['replay_size']
    batch_size = int(32)
    observation_period = params['replay_initial']
    learning_rate = params['learning_rate']
    # ----------------------------------------
    render = True
    #render = False
    # Define environment/game
    env_name = params['env_name']
    env = make_env(env_name)
    observation = env.reset()
    print env.observation_space.shape
    # ----------------------------------------
    # Actions
    # Type: Discrete(4)
    # Num | Observation |
    # 0   | 'NOOP'      |
    # 1   | 'FIRE'      |
    # 2   | 'RIGHT'     |
    # 3   | 'LEFT'      |
    n_action = env.action_space.n
    actions = np.array([0, 1, 2, 3])
    # ----------------------------------------
    # Observation
    # Type: Box(210, 160, 3) -> (84x84)
    input_size = env.observation_space.shape
    # ----------------------------------------
    # Initialize Neural DQN object
    AI = DeepQLearner(input_size, actions, hidden_neurons, batch_size, update_target, epsilon, gamma, learning_rate)
    # Load pre-trained model
    AI.importNetwork('models/%s_Deep_Q_network_epoch_20000' % (env_name))
    observation = env.reset()
    env.render()
    raw_input('Press enter to start:')
    # ----------------------------------------
    # Test
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        action = actions[0]
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            if render: env.render()

            # Epsilon-Greedy policy
            #action = AI.eGreedy(observation.__array__())
            action = AI.greedy(observation.__array__())

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)

            #
            step += 1
            total_reward += reward

        # End of the single episode training
        if total_reward > params['stop_reward']: break

        #R_plot.append([e, total_reward])
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e, total_reward, step, AI.epsilon))
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF
