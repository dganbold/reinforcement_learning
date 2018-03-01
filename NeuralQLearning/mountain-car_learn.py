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
    gamma = 0.98
    epoch = 1001
    max_steps = 200
    # Define parameters for Q-network
    hidden_neurons = 50
    update_target = 100
    max_memory = max_steps*10
    batch_size = int(32)
    #
    render = False
    # ----------------------------------------
    # Define environment/game
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation
    # 0   | push_left
    # 1   | no_push
    # 2   | push_right
    n_action = env.action_space.n
    actions = np.array([0, 1, 2])
    # ----------------------------------------
    # Observation
    # Type: Box(2)
    # Num | Observation | Min   | Max
    # 0   | position    | -1.2  | 0.6
    # 1   | velocity    | -0.07 | 0.07
    n_input = env.observation_space.shape[0]
    observation = []
    #state_scale = 1.0/(env.observation_space.high - env.observation_space.low)
    #state_offset = env.observation_space.low
    # ----------------------------------------
    # Initialize Plot object
    R_plot = PlotTimeSeries(x_lim=[0, 10], y_lim=[-max_steps-10, 0], size=(6.8, 5.0))
    R_plot.create([list(), list()], 'Episode', 'Total Reward', 'Neural Q-Learning: ' + env_name)
    #
    titles = "Actions " + u"\u25C0" + ":push_left/" + u"\u25AA" + ":no_push/" + u"\u25B6" + ":push_right"
    Q_plot = Plot2D(x_lim=[-1.2, 0.6], y_lim=[-0.07, 0.07], x_n=27, y_n=27, size=(5.8, 5.0))
    S_mesh = Q_plot.getMeshGrid()
    #S_mesh = state_scale * (S_mesh - state_offset)
    # ----------------------------------------
    # Initialize Neural Q-Learner object
    AI = NeuralQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, gamma)
    Q_mesh = AI.evaluate_Q_target(S_mesh)
    # Plot
    Q_plot.create(Q_mesh, 'Position', 'Velocity', titles)
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        #state = env.reset()
        #Take norm
        #state = state_scale * (observation - state_offset)
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            state_capture = observation.copy()
            #state_capture = state.copy()
            # Render
            if render: env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(state_capture)

            # Apply action, get rewards and new state
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.6:
                reward = 1
                game_over = True
            else:
                reward += 5.*np.abs(observation[1])

            step += 1
            if step >= max_steps and done:
                game_over = True

            # Take norm
            #state = state_scale * (observation - state_offset)
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
            total_reward += reward
        # End of the single episode training
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e, total_reward, step, AI.epsilon))
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        # Plot
        R_plot.append([e, total_reward])
        Q_mesh = AI.evaluate_Q_target(S_mesh)
        Q_plot.update(Q_mesh)
        # Export trained Q-Network
        if (e % 50) == 0 and e > 500:
            AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, e))
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
