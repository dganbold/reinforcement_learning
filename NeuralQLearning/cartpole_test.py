#
#
import gym
from NeuralQLearner import *
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
    epoch = 10
    max_steps = 200
    # Define parameters for Q-network
    hidden_neurons = 50
    update_target = 128
    max_memory = max_steps*10
    batch_size = int(32)
    #
    render = True
    # ----------------------------------------
    # Define environment/game
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    # ----------------------------------------
    # Actions
    # Type: Discrete(2)
    # Num | Observation
    # 0   | Push cart to the left
    # 1   | Push cart to the right
    n_action = env.action_space.n
    actions = np.array([0, 1])
    # ----------------------------------------
    # Observation
    # Type: Box(4)
    # Num | Observation   | Min    | Max
    # 0   | Cart Position | -2.4   | 2.4
    # 1   | Cart Velocity | -Inf   | Inf
    # 2   | Pole Angle    | -41.8  | 41.8
    # 3   | Pole Velocity | -Inf   | Inf
    n_input = env.observation_space.shape[0]
    observation = []
    # ----------------------------------------
    # Initialize Neural Q-Learner object
    AI = NeuralQLearner(n_input, actions, hidden_neurons, batch_size, update_target, epsilon, gamma)
    # Load pre-trained model
    AI.importNetwork('models/%s_Q_network_epoch_500' % (env_name))
    # ----------------------------------------
    # Test
    observation = env.reset()
    env.render()
    raw_input('Press enter to start:')
    for e in range(epoch):
        # Get initial input
        observation = env.reset()

        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while not game_over:
            state_capture = observation.copy()
            if render: env.render()

            # Greedy policy
            action = AI.greedy(state_capture)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)
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
