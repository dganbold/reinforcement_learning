#
#
import gym
from NeuralQLearner import *
from PlotWrapper import *
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = 0.5   # exploration
    epsilon_floor = 0.05
    exploration_decay = 0.998
    # Define parameters for Q-learning
    alpha = 0.2
    gamma = 0.98
    epoch = 1000
    max_steps = 200
    # Define parameters for Q-network
    batch_size = 32
    hidden_neurons = 50
    update_target = 100
    max_memory = 2000
    #
    render = False
    # ----------------------------------------
    # Define environment/game
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation  | Min   | Max
    # 0   | Joint effort | -2.0  | 2.0
    n_action = 21
    actions = np.linspace(-2, 2, num=n_action, endpoint=True)
    # ----------------------------------------
    # Observation
    # Type: Box(2)
    # Num | Observation | Min   | Max
    # 0   | cos(theta)  | -1.0  | 1.0
    # 1   | sin(theta)  | -1.0  | 1.0
    # 2   | theta dot   | -8.0  | 8.0
    n_input = env.observation_space.shape[0]
    observation = []
    # ----------------------------------------
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 100], y_lim=[-1600, 0], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    # ----------------------------------------
    # Initialize Neural Q-Learner object
    AI = NeuralQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, alpha, gamma)
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        observation_init = observation

        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while (not game_over):
            observation_capture = observation
            if render: env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(observation)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step([action])

            # Store experience
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            experienceMemory.memorize([observation_capture, action, reward, observation], game_over)

            # Recall and replay experience
            miniBatch = experienceMemory.recall(batch_size)
            # Refinement of model
            if len(miniBatch) == batch_size:
                AI.train_Q_network(miniBatch)
            #
            step += 1
            total_reward += reward
        # End of the single episode training
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e, total_reward, step, AI.epsilon))
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        # Plot
        R_plot.append([e, total_reward])
        #
    # ----------------------------------------
    # Export trained Neural-Net
    AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, epoch))
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF
