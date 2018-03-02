
import gym
from DeepQLearner import *
from PlotWrapper import *
#
#
def preprocess(input_observation):
  # prepro 210x160x3 uint8 frame into 6400 (80x80)
  # 1D location float vector
  I = input_observation[35:195]    # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0  # erase background (background type 1)
  I[I == 109] = 0  # erase background (background type 2)
  I[I != 0] = 1    # everything else (paddles, ball) just set to 1
  # Convert from 80 x 80 matrix to 1600 x 1 matrix
  #return I.astype(np.float).ravel()
  return I.astype(np.float)
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = 1.0   # exploration
    epsilon_floor = 0.1
    exploration_decay = 0.998
    # Define parameters for Q-learning
    gamma = 0.98
    epoch = 2000
    # Define parameters for Q-network
    hidden_neurons = 32
    update_target = 100
    max_memory = 10000
    batch_size = int(32)
    #
    render = True
    # ----------------------------------------
    # Define environment/game
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    observation_raw = env.reset()
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
    actions = np.array([0, 1, 2, 3, 4, 5])
    # ----------------------------------------
    # Observation
    # Type: Box(210, 160, 3) -> 6400 (80x80)
    observation = preprocess(observation_raw)
    # Initialize frame buffer object
    m = 4
    frameBuffer0 = FIFO(m)
    frameBuffer1 = FIFO(m)
    for n in range(m):
        frameBuffer0.push(observation)
        frameBuffer1.push(observation)
    #
    n_input = frameBuffer0.pop_stack().shape
    # ----------------------------------------
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 10], y_lim=[-25, 25], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    # ----------------------------------------
    # Initialize Neural DQN object
    AI = DeepQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, gamma)
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation_raw = env.reset()
        action = actions[0]

        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            frameBuffer0.push(preprocess(observation_raw.copy()))
            if render: env.render()

            # Epsilon-Greedy policy
            if (step % m) == 0 and step > 0:
                action = AI.eGreedy(frameBuffer0.pop_stack())

            # Apply action, get rewards and new state
            observation_raw, reward, game_over, info = env.step(action)
            frameBuffer1.push(preprocess(observation_raw.copy()))

            step += 1
            # Store experience
            if (step % m) == 0 or game_over:
                # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
                experienceMemory.memorize([frameBuffer0.pop_stack(), action, reward, frameBuffer1.pop_stack()], game_over)
                # Recall and replay experience
                miniBatch = experienceMemory.recall(batch_size)
                # Refinement of model
                if len(miniBatch) == batch_size:
                    AI.train_Q_network(miniBatch)
            #
            total_reward += reward
        # End of the single episode training
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e, total_reward, step, AI.epsilon))
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        # Plot
        R_plot.append([e, total_reward])
        # Export trained Q-Network
        if (e % 50) == 0 and e > 100:
            AI.exportNetwork('models/%s_Deep_Q_network_epoch_%d' % (env_name, e))
        #
    # ----------------------------------------
    # Export trained Q-Network
    #AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, epoch))
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF

#imgplot = plt.imshow(frame_stack0)
#plt.draw()
#plt.pause(0.5)
