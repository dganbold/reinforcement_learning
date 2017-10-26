#
#
import gym
from TabularQLearner import *
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
    epoch = 1000
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
    AI = TabularQLearner(actions,epsilon=epsilon,alpha=alpha, gamma=gamma)
    # Load pre-trained model
    #AI.importQ('Q_table_27_27_3_epoch_1000')
    AI.plotQ()
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
                #reward += 0.1*np.abs(observation_init[0]-observation[0]) + 10.*np.abs(observation[1]) #+ observation[0]/5.

            step += 1
            if (step >= max_steps or observation[0] >= 0.6) and done:
                game_over = True

            # Refinement of model
            AI.learnQ(observation_capture, action, reward, observation)
        #
        if reward > 0:
            print("#TRAIN Episode:{} finished after {} timesteps. Reached GOAL!.".format(e,step))
        else:
            print("#TRAIN Episode:{} finished after {} timesteps.".format(e,step))
        # Update plot
        AI.plotQupdate()
        # Update exploration
        AI.epsilon *= exploration_decay
        AI.epsilon = max(epsilon_floor, AI.epsilon)
        #
    # ----------------------------------------
    # Export Q table
    AI.exportQ('Q_table_%d_%d_3_epoch_%d' % (AI.N_position,AI.N_velocity,epoch))
    AI.plotQaction()
    # ----------------------------------------
    print("Done!.")
    # Close environment
    env.close
    # Some delay
    raw_input('Press enter to terminate:')
