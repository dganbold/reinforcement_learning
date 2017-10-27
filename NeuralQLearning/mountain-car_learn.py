#
#
import gym
from NeuralQLearner import *
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for greedy policy
    epsilon = 0.5   # exploration
    epsilon_floor = 0.1
    exploration_decay = 0.998
    # Define parameters for Q-learning
    alpha = 0.2
    gamma = 0.98
    epoch = 1000
    max_steps = 1000
    max_memory = max_steps*10
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
    env = gym.make('MountainCar-v0')
    # ----------------------------------------
    # Initialize Neural Q-Learn object
    AI = NeuralQLearner(n_input, actions, batch_size, epsilon, alpha, gamma)
    AI.plotQ()
    # Initialize experience replay object
    exp = Experience(max_memory)
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        observation_init = observation

        # Training for single episode
        step = 0
        reward = -1
        game_over = False
        while (not game_over):
            observation_capture = observation
            #env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(observation)

            # Apply action, get rewards and new state
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.6:
                reward = 1
            else:
                reward += 10.*np.abs(observation[1])

            step += 1
            if (step >= max_steps or observation[0] >= 0.6) and done:
                game_over = True

            # Store experience
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            exp.memorize([observation_capture, action, reward, observation], game_over)

            # Recall and replay experience
            miniBatch = exp.recall(batch_size)
            # Refinement of model
            if len(miniBatch) == batch_size:
                AI.train_Q_network(miniBatch)

        if max_steps > step:
            print("#TRAIN Episode:{} finished after {} timesteps. exploration:{}. Reached GOAL!.".format(e, step, AI.epsilon))
        else:
            print("#TRAIN Episode:{} finished after {} timesteps. exploration:{}".format(e, step, AI.epsilon))
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        # Plot
        AI.plotQupdate()
        #
    # ----------------------------------------
    # Export trained Neural-Net
    AI.exportNetwork('Q_network_epoch_%d' % (epoch))
    # ----------------------------------------
    print("Done!.")
    # Close environment
    env.close()
    # Some delay
    raw_input('Press enter to terminate:')
