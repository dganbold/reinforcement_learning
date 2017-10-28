#
#
import gym
import random
import numpy as np
import pygame
import time
from pygame.locals import *
#
# sudo ds4drv --led ff0000
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters
    epoch = 10
    max_steps = 1000
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation  | Min   | Max
    # 0   | Joint effort | -2.0  | 2.0
    n_action = 1
    action = 0
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
    # Initialize the joysticks
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    #print len(joysticks)
    joystick = joysticks[0]
    joystick.init()
    name = joystick.get_name()
    print("Joystick name: {}".format(name))
    axes = joystick.get_numaxes()
    print("Number of axes: {}".format(axes))
    buttons = joystick.get_numbuttons()
    print("Number of buttons: {}".format(buttons))
    hats = joystick.get_numhats()
    print("Number of hats: {}".format(hats) )
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()
        observation_init = observation
        env.render()
        # ----------------------------------------
        # Wait for start [o] key
        print('Press [o] key to start episode...')
        while (True):
            try:
                pygame.event.get()
                if joystick.get_button(2):
                    print("[o] is pressed!.")
                    break
                elif joystick.get_button(1):
                    print("[x] is pressed!.Exit")
                    # Close environment
                    env.close()
                    pygame.quit()
                    exit(0)
            except KeyboardInterrupt:
                print("KeyboardInterrupt!.")
                pygame.quit()
                exit(0)
        # ----------------------------------------
        # Training for single episode
        step = 0
        total_reward = 0
        game_over = False
        while (not game_over):
            observation_capture = observation
            env.render()

            # Human policy
            try:
                pygame.event.get()
                #action = 1
                if joystick.get_button(1):
                    print("[x] is pressed!.Exit")
                    pygame.quit()
                    exit(0)
                action = [2*joystick.get_axis(0)]
            except KeyboardInterrupt:
                print("KeyboardInterrupt!.")
                pygame.quit()
                exit(0)

            # Apply action, get rewards and new state
            observation, reward, done, info = env.step(action)

            step += 1
            if (step >= max_steps) and done:
                game_over = True
            #
            total_reward += reward
    	    # Used to manage how fast the screen updates
            time.sleep(0.01)
        # End of the single episode training
        print('#TRAIN Episode:%3i, Reward:%7.3f, Steps:%3i'%(e, total_reward, step))
        #
    # ----------------------------------------
    pygame.quit()
    print("Done!.")
    # Some delay
    time.sleep(2)
