#
#
import gym
from NeuralQLearner import *
from PlotWrapper import *
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for greedy policy
    epsilon = 0.5   # exploration
    epsilon_floor = 0.05
    exploration_decay = 0.998
    # Define parameters for Q-learning
    gamma = 0.98
    epoch = 1001
    max_steps = 500
    # Define parameters for Q-network
    hidden_neurons = 50
    update_target = 100
    max_memory = max_steps*5
    batch_size = int(32)
    #
    render = False
    # Define environment/game
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation
    # 0   | Joint effort (TORQUE -1)
    # 1   | Joint effort (TORQUE  0)
    # 2   | Joint effort (TORQUE  1)
    n_action = 3
    actions = np.array([0, 1, 2])
    # ----------------------------------------
    # Observation
    # Type: Box(4)
    # Num | Observation   | Min    | Max
    # 0   | Cart Position | -2.4   | 2.4
    # 1   | Cart Velocity | -Inf   | Inf
    # 2   | Pole Angle    | -41.8  | 41.8
    # 3   | Pole Velocity | -Inf   | Inf
    # 4   | Pole Velocity | -Inf   | Inf
    # 5   | Pole Velocity | -Inf   | Inf
    n_input = 6
    observation = []
    # ----------------------------------------
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 20], y_lim=[-max_steps-20, 0], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    #
    # ----------------------------------------
    # Initialize Neural Q-Learn object
    AI = NeuralQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, gamma)
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()

        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while (not game_over):
            state_capture = observation.copy()
            if render: env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(state_capture)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)
            state = observation.copy()
            # Store experience
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            experienceMemory.memorize([state_capture, action, reward, state], game_over)

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
        # Export trained Q-Network
        if (e % 50) == 0 and e > 500:
            AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, e))
        #
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()