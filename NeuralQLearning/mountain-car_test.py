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
    max_steps = 500
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
    env = gym.make('MountainCar-v0')
    # ----------------------------------------
    # Initialize QLearn object
    AI = NeuralQLearner(n_input, actions, epsilon=epsilon)
    # Load pre-trained model
    AI.importNetwork('Q_network_epoch_1000')
    AI.plotQ()
    AI.plotQaction()
    # ----------------------------------------
    # Test
    for e in range(epoch):
        # Clear trajectory
        AI.clearTrajectory()
        plt.pause(0.001)

        # Get initial input
        observation = env.reset()
        # Training for single episode
        step = 0
        game_over = False
        while (not game_over):
            observation_capture = observation
            env.render()

            # Greedy policy
            action = AI.greedy(observation)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)

            step += 1
            # Plot trajectory
            AI.plotTrajectory(observation_capture, action)
        #
        if observation[0] > 0.5:
            print("#TEST Episode:{} finished after {} timesteps. Reached GOAL!.".format(e, step))
        else:
            print("#TEST Episode:{} finished after {} timesteps. Timeout!.".format(e, step))
        #
        # Plot
        plt.pause(1.5)
    # ----------------------------------------
    print("Done!.")
    # Close environment
    env.close
    # Some delay
    raw_input('Press enter to terminate:')
