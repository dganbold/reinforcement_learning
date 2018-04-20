
import gym
from DeepQLearner import *
from PlotWrapper import *
import atari_wrappers
#
#
def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
    return env
#
#
if __name__ == "__main__":
    # ----------------------------------------
    params = atari_wrappers.HYPERPARAMS['pong']
    # ----------------------------------------
    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']
    epsilon_floor = params['epsilon_final']
    exploration_slope = (params['epsilon_final'] - params['epsilon_start'])/params['epsilon_frames']
    # Define parameters for Q-learning
    gamma = params['gamma'] #0.98
    epoch = 10
    # Define parameters for Q-network
    update_target = params['target_net_sync']
    max_memory = params['replay_size']
    batch_size = int(32)
    observation_period = params['replay_initial']
    learning_rate = params['learning_rate']
    #
    render = True
    #render = False
    # ----------------------------------------
    # Define environment/game
    env_name = params['env_name']
    env = make_env(params)
    observation = env.reset()
    print env.observation_space.shape
    # ----------------------------------------
    # Actions
    # Type: Discrete(4)
    # Num | Observation |
    # 0   | 'NOOP'      |
    # 1   | 'FIRE'      |
    # 2   | 'RIGHT'     |
    # 3   | 'LEFT'      |
    # 4   | 'RIGHTFIRE' |
    # 5   | 'LEFTFIRE'  |
    actions = np.array([0, 1, 2, 3, 4, 5])
    n_action = env.action_space.n
    # ----------------------------------------
    # Observation
    # Type: Box(210, 160, 3) -> (84x84)
    input_size = env.observation_space.shape
    # ----------------------------------------
    # Initialize Neural DQN object
    AI = DeepQLearner(input_size, actions, batch_size, update_target, epsilon, gamma, learning_rate)
    # Load pre-trained model
    AI.importNetwork('models/%s_NIPS_Deep_Q_network_epoch_1000_Adam' % (env_name))
    # Initialize experience replay object
    experienceMemory = Experience(max_memory)
    #
    total_steps = 0
    max_reward = -float('inf')
    # ----------------------------------------
    # Train
    for e in range(1,epoch+1):
        # Get initial input
        observation = env.reset()
        action = actions[0]
        # Training for single episode
        steps = 0
        total_reward = 0
        game_over = False
        while not game_over:
            state_capure = observation.__array__().copy()
            if render: env.render()

            # Epsilon-Greedy policy
            #action = AI.eGreedy(observation.__array__())
            action = AI.greedy(observation.__array__())

            # Apply action, get rewards and new state
            observation, reward, game_over, info = env.step(action)
            state = observation.__array__().copy()

            #
            steps += 1
            total_reward += reward

        # End of the single episode training
        total_steps += steps
        print('#TRAIN Episode:%4i, Reward:%3.2f, Steps:%4i, Exploration:%1.4f, Frames:%6i'%(e, total_reward, steps, AI.epsilon, total_steps))
        #
    # ----------------------------------------
    print("Done!.")
    # Some delay
    raw_input('Press enter to terminate:')
    # Close environment
    env.close()
# EOF
