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
    epsilon_floor = 0.1
    exploration_decay = 0.998
    # Define parameters for Q-learning
    alpha = 0.2
    gamma = 0.98
    epoch = 10
    max_steps = 500
    # Define parameters for Q-network
    batch_size = 128
    hidden_neurons = 50
    update_target = 32
    max_memory = 10000
    #
    render = True
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
    # ----------------------------------------
    # Initialize Plot object
    titles = "Actions " + u"\u25C0" + ":push_left/" + u"\u25AA" + ":no_push/" + u"\u25B6" + ":push_right"
    Q_plot = Plot2D(x_lim=[-1.2, 0.6], y_lim=[-0.07, 0.07], x_n=27, y_n=27, size=(5.8, 5.0))
    S_mesh = Q_plot.getMeshGrid()
    # ----------------------------------------
    # Initialize Neural Q-Learner object
    AI = NeuralQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, alpha, gamma)
    # Load pre-trained model
    AI.importNetwork('models/%s_Q_network_epoch_1000' % (env_name))
    # Evaluate Q-network and plot
    Q_mesh = AI.evaluate_Q_target(S_mesh)
    Q_plot.create(Q_mesh, 'Position', 'Velocity', titles)
    # ----------------------------------------
    # Test
    observation = env.reset()
    env.render()
    raw_input('Press enter to start:')
    for e in range(epoch):
        # Clear trajectory
        Q_plot.deletePoints()

        # Get initial input
        observation = env.reset()
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            observation_capture = observation
            if render: env.render()

            # Greedy policy
            action = AI.greedy(observation)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)

            # Plot trajectory
            Q_plot.appendPoint(observation_capture)
            #
            step += 1
            total_reward += reward
        # End of the single episode testing
        print('#TEST Episode:%2i, Reward:%7.3f, Steps:%3i' % (e, total_reward, step))
        # Plot
        plt.pause(1.5)
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF
