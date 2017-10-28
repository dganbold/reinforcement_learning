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
    max_steps = 200
    max_memory = max_steps*10
    batch_size = int(32)
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
    n_input = 3
    observation = []
    # ----------------------------------------
    # Define environment/game
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    # ----------------------------------------
    # Initialize Neural Q-Learn object
    AI = NeuralQLearner(n_input, actions, batch_size, epsilon, alpha, gamma)
    #AI.plotQ()
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
        total_reward = 0
        game_over = False
        while (not game_over):
            observation_capture = observation
            #env.render()

            # Epsilon-Greedy policy
            action = AI.eGreedy(observation)

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step([action])

            # Store experience
            # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
            exp.memorize([observation_capture, action, reward, observation], game_over)

            # Recall and replay experience
            miniBatch = exp.recall(batch_size)
            # Refinement of model
            if len(miniBatch) == batch_size:
                AI.train_Q_network(miniBatch)
            #
            step += 1
            total_reward += reward
        #  End of the One-Episode training
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i, Exploration:%1.4f'%(e, total_reward, step, AI.epsilon))
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        # Plot
        #AI.plotQupdate()
        #
    # ----------------------------------------
    # Export trained Neural-Net
    AI.exportNetwork('models/%s_Q_network_epoch_%d' % (env_name, epoch))
    # ----------------------------------------
    print("Done!.")
    # Close environment
    env.close()
    # Some delay
    raw_input('Press enter to terminate:')
