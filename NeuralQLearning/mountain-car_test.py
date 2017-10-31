#
#
import gym
from NeuralQLearner import *
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = 1.0   # exploration
    epsilon_floor = 0.1
    exploration_decay = 0.995
    # Define parameters for Q-learning
    alpha = 0.2
    gamma = 0.97
    epoch = 10
    batch_size = int(32)
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation
    # 0   | push_left
    # 1   | no_push
    # 2   | push_right
    n_action = 3
    actions = [0, 1, 2]
    # ----------------------------------------
    # Observation
    # Type: Box(2)
    # Num | Observation | Min   | Max
    # 0   | position    | -1.2  | 0.6
    # 1   | velocity    | -0.07 | 0.07
    n_input = 2
    observation = []
    # ----------------------------------------
    # Define environment/game
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    # ----------------------------------------
    # Initialize QLearn object
    AI = NeuralQLearner(n_input, actions, batch_size, epsilon, alpha, gamma)
    # Load pre-trained model
    AI.importNetwork('models/%s_Q_network_epoch_1000' % (env_name))
    # Plot Eval-Q
    AI.plotQ()
    AI.plotQaction()
    # ----------------------------------------
    # Test
    env.render()
    raw_input('Press enter to start:')
    for e in range(epoch):
        # Clear trajectory
        AI.clearTrajectory()
        plt.pause(0.001)

        # Get initial input
        observation = env.reset()
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while (not game_over):
            observation_capture = observation
            env.render()

            # Greedy policy
            action = AI.greedy(observation)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)

            # Plot trajectory
            AI.plotTrajectory(observation_capture, action)
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
