
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
    #BreakoutNoFrameskip-v4
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']   # exploration
    epsilon_floor = params['epsilon_final'] #0.05
    exploration_decay = 0.9999
    # Define parameters for Q-learning
    gamma = params['gamma'] #0.98
    epoch = 40000
    # Define parameters for Q-network
    hidden_neurons = 256
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
    #env_name = 'Pong-v0'
    env_name = params['env_name']
    #env = gym.make(env_name)
    env = make_env(params)
    observation = env.reset()
    print env.observation_space.shape
    # ----------------------------------------
    # Actions
    # Type: Discrete(6)
    # Num | Observation |
    # 0   | 'NOOP'      |
    # 1   | 'FIRE'      |
    # 2   | 'RIGHT'     |
    # 3   | 'LEFT'      |
    # 4   | 'RIGHTFIRE' |
    # 5   | 'LEFTFIRE'  |
    n_action = env.action_space.n
    #print n_action
    actions = np.array([0, 1, 2, 3])
    #actions = np.array([0, 1, 2, 3, 4, 5])
    #print n_action
    # ----------------------------------------
    # Observation
    # Type: Box(210, 160, 3) -> (84x84)
    input_size = env.observation_space.shape
    # ----------------------------------------
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 10], y_lim=[-25, 25], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    # ----------------------------------------
    # Initialize Neural DQN object
    AI = DeepQLearner(input_size, actions, hidden_neurons, batch_size, update_target, epsilon, gamma, learning_rate)
    # Load pre-trained model
    AI.importNetwork('models/%s_Deep_Q_network_epoch_42000' % (env_name))
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    #
    max_total_reward = 0
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        action = actions[0]
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            state_capure = observation.__array__().copy()
            if render: env.render()

            # Epsilon-Greedy policy
            #if (step % m) == 0 and step > 0:
            action = AI.eGreedy(observation.__array__())

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)
            state = observation.__array__().copy()

            # Store experience
            #done = game_over
            #if reward == 0: done = 0
            #else:           done = 1
            #if done == 1 and game_over == 1: print 'Game over!'
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            experienceMemory.memorize([state_capure, action, reward, state], game_over)

            # Refinement of model
            #if experienceMemory.getsize() > observation_period:
            miniBatch = experienceMemory.recall(batch_size)
            if len(miniBatch) == batch_size:
                AI.train_Q_network(miniBatch)
            #
            step += 1
            total_reward += reward

        # End of the single episode training
        if total_reward > params['stop_reward']: break

        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        max_total_reward = max(max_total_reward,total_reward)
        # Plot
        if (e % 10) == 0:
            R_plot.append([e, max_total_reward])
            print('#TRAIN Episode:%3i, Max Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e+42000, max_total_reward, step, AI.epsilon))
            max_total_reward = 0

        # Export trained Q-Network
        if (e % 1000) == 0 and e >= 1000:
            # Dump reward to csv
            R_plot.export('models/%s_Deep_Q_network_epoch_%d_rewards.csv' % (env_name, e+42000))
            AI.exportNetwork('models/%s_Deep_Q_network_epoch_%d' % (env_name, e+42000))
        #
    # ----------------------------------------
    # Export trained Q-Network
    #AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, epoch))
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Dump reward to csv
    R_plot.export('%s_Deep_Q_network_epoch_%d_rewards.csv' % (env_name, e+42001))
    # Close environment
    env.close()
# EOF
