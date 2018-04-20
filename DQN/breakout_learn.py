
import gym
from DeepQLearner import *
from PlotWrapper import *
import atari_wrappers
#
#
def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
    return env
#
#
if __name__ == "__main__":
    # ----------------------------------------
    params = atari_wrappers.HYPERPARAMS['breakout-small']
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']
    epsilon_floor = params['epsilon_final']
    exploration_slope = (params['epsilon_final'] - params['epsilon_start'])/params['epsilon_frames']
    # Define parameters for Q-learning
    gamma = params['gamma'] #0.98
    epoch = 50000
    # Define parameters for Q-network
    update_target = params['target_net_sync']
    max_memory = params['replay_size']
    batch_size = int(32)
    observation_period = params['replay_initial']
    learning_rate = params['learning_rate']
    #
    #render = True
    render = False
    # ----------------------------------------
    # Define environment/game
    env_name = params['env_name']
    env = make_env(params)
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
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 10], y_lim=[-5, 50], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    # ----------------------------------------
    # Initialize Neural DQN object
    AI = DeepQLearner(input_size, actions, batch_size, update_target, epsilon, gamma, learning_rate)
    # Load pre-trained model
    #AI.importNetwork('models/%s_Deep_Q_network_epoch_20000' % (env_name))
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    #
    total_steps = 0
    max_reward = -float('inf')
    # ----------------------------------------
    # Train
    for e in range(1,epoch+1):
        # Get initial input
        observation = env.reset()
        action = actions[0]
        # Training for single episode
        steps = 0
        total_reward = 0
        game_over = False
        while not game_over:
            state_capure = observation.__array__().copy()
            if render: env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(observation.__array__())

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)
            state = observation.__array__().copy()

            # Store experience
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            experienceMemory.memorize([state_capure, action, reward, state], game_over)

            # Refinement of model
            #if len(miniBatch) == batch_size:
            if experienceMemory.getsize() > observation_period:
                miniBatch = experienceMemory.recall(batch_size)
                AI.train_Q_network(miniBatch)
            #
            steps += 1
            total_reward += reward

        # End of the single episode training
        total_steps += steps
        print('#TRAIN Episode:%4i, Reward:%3.2f, Steps:%3i, Exploration:%1.4f, Frames:%6i'%(e, total_reward, steps, AI.epsilon, total_steps))
        if total_reward > params['stop_reward']: break

        # Update exploration
        AI.epsilon = exploration_slope*total_steps + params['epsilon_start']
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        #max_reward = max(max_reward,total_reward)
        # Plot
        #if (e % 10) == 0:
        R_plot.append([e, total_reward])
        #max_reward = -float('inf')

        # Export trained Q-Network
        if (e % 1000) == 0:
            # Dump reward to csv
            R_plot.export('models/%s_NIPS_Deep_Q_network_epoch_%d_rewards.csv' % (env_name, e))
            # Export trained Q-Network
            AI.exportNetwork('models/%s_NIPS_Deep_Q_network_epoch_%d' % (env_name, e))
        #
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF
