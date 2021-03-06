# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from args import get_args

import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, axs = plt.subplots(1, 1, figsize=(5.8, 5))
#plt.ion()
#plt.show(block=False)


class Experience(object):

    def __init__(self, max_memory=1000, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def memorize(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def recall(self, batch_size=32):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        indices = random.sample(np.arange(len(self.memory)), min(batch_size, len(self.memory)))
        batch = []
        for index in indices:
            batch.append(self.memory[index])
        return batch

    # END Experience class


class NeuralQLearner:
    def __init__(self, n_states, actions, batch_size=int(128), epsilon=0.1, alpha=0.2, gamma=0.9):
        # Get context.
        from nnabla.contrib.context import extension_context
        args = get_args()
        print "weight_decay:", args.weight_decay
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        logger.info("Running in %s" % extension_module)
        ctx = extension_context(extension_module, device_id=args.device_id)
        nn.set_default_context(ctx)

        # Q-Learing parametes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.actions = actions
        self.n_actions = len(actions)
        self.n_states = n_states

        # Neural network's training parametes
        self.learning_rate = 1e-3
        self.batch_size = batch_size
        self.model_save_path = 'models'
        self.model_save_interval = 1000
        self.weight_decay = 0

        # State-Action Plot's parametes
        self.plim = [-1.2, 0.6]
        self.vlim = [-0.07, 0.07]
        self.N_position = 27
        self.N_velocity = 27
        self.positions = np.linspace(self.plim[0], self.plim[1], num=self.N_position, endpoint=True)
        self.velocities = np.linspace(self.vlim[0], self.vlim[1], num=self.N_velocity, endpoint=True)

        # --------------------------------------------------
        print "Initializing the Neural Network."
        # --------------------------------------------------
        # Hidden layer's neuron number
        hn = 50
        # Preparing the Computation Graph for Q
        self.Q_x = nn.Variable([self.batch_size, self.n_states])
        self.Q_y = nn.Variable([self.batch_size, self.n_actions])

        # Construct Q-Network for Q-Learning.
        l1 = F.tanh(PF.affine(self.Q_x, hn, name='affine1'))
        self.Q_Network = PF.affine(l1, self.n_actions, name='affine2')
        self.Q_Network.persistent = True

        # Create loss function.
        #self.loss = F.mean(F.squared_error(self.train_model, self.yt))
        self.loss = F.mean(F.huber_loss(self.Q_Network, self.Q_y))

        # Preparing the Computation Graph for target Q-Network
        self.Q_target_x = nn.Variable([self.batch_size, self.n_states])
        self.Q_target_w1 = nn.Variable([self.n_states, hn], need_grad=False)   # Weights
        self.Q_target_b1 = nn.Variable([hn], need_grad=False)                  # Biases
        self.Q_target_w2 = nn.Variable([hn, self.n_actions], need_grad=False)  # Weights
        self.Q_target_b2 = nn.Variable([self.n_actions], need_grad=False)      # Biases

        # Construct target Q-Network for Q-Learning.
        h1 = F.tanh(F.affine(self.Q_target_x , self.Q_target_w1, self.Q_target_b1))
        self.Q_target_Network = F.affine(h1, self.Q_target_w2, self.Q_target_b2)
        self.update_Q_target()

        # --------------------------------------------------
        print "Initializing the Solver."
        # --------------------------------------------------
        # Create Solver
        # self.solver = S.Sgd(self.learning_rate)
        self.solver = S.RMSprop(self.learning_rate, 0.95)
        self.solver.set_parameters(nn.get_parameters())

        self.update_Q = 100
        self.iter = 0
        #
        self.plot_reset = True

    def train_Q_network(self, input):
        # Perform a minibatch Q-learning update
        # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # ----------------------------------------------
        # Preparing data
        s0_vector = np.zeros((self.batch_size, self.n_states))
        s1_vector = np.zeros((self.batch_size, self.n_states))
        for n in range(self.batch_size):
            s0_vector[n] = input[n][0][0]  # state_t
            s1_vector[n] = input[n][0][3]  # state_t+1
        # ----------------------------------------------
        # Prediction of Q-Value on current state.
        self.Q_x.d = s0_vector.copy()
        self.Q_Network.forward(clear_buffer=True)
        Q_present = self.Q_Network.d.copy()
        # ----------------------------------------------
        # Prediction of Q-Value at next state.
        self.Q_target_x.d = s1_vector.copy()
        self.Q_target_Network.forward(clear_buffer=True)
        Q_next = self.Q_target_Network.d.copy()
        maxQ = np.amax(Q_next, axis=1)
        #maxQ = np.reshape(maxQ, (-1, 1))
        # ----------------------------------------------
        # Calculate target value of Q-network
        target = Q_present.copy()
        for n in range(self.batch_size):
            a = np.argmin(abs(self.actions - input[n][0][1] ), axis=0)
            #a = input[n][0][1]  # action_t
            if input[n][1]:     # game_over
                target[n][a] = input[n][0][2]  # reward_t
            else:
                target[n][a] = input[n][0][2] + self.gamma * maxQ[n]
        # ----------------------------------------------
        # Training forward
        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # DQN: target = reward(s,a) + gamma * max(Q(s'))
        self.Q_x.d = s0_vector.copy()
        self.Q_y.d = target.copy()
        # Forward propagation given inputs.
        #self.solver.zero_grad()
        self.loss.forward(clear_no_need_grad=True)
        # Parameter gradients initialization and gradients computation by backprop.
        # Initialize grad
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # Apply weight decay and update by Adam rule.
        self.solver.weight_decay(self.weight_decay)
        self.solver.update()
        #mse = training_error(Q_Network.d, Q_y.d)
        self.iter += 1

        # Every C updates clone the Q-network to target Q-network
        if self.iter % self.update_Q == 0:
            #print "Updating target Q-network"
            self.update_Q_target()

    def update_Q_target(self):
        params = nn.get_parameters().items()
        self.Q_target_w1.d = params[0][1].d.copy()
        self.Q_target_b1.d = params[1][1].d.copy()
        self.Q_target_w2.d = params[2][1].d.copy()
        self.Q_target_b2.d = params[3][1].d.copy()

    def training_error(self, y_pred, y_taget):
        mse = ((y_pred - y_taget) ** 2).sum(axis=1).mean()
        return mse

    def eGreedy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.greedy(state)
        return action

    def greedy(self, state):
        self.Q_x.d = state
        self.Q_Network.forward(clear_buffer=True)
        Q = self.Q_Network.d.copy()
        maxQ = np.max(Q[0])
        best = [i for i in range(len(self.actions)) if Q[0][i] == maxQ]
        if len(best) > 1:
            i = random.choice(best)
        else:
            i = np.argmax(Q[0])
        return self.actions[i]

    def exportNetwork(self, fname):
        print("Export Q-network to {}".format(fname))
        nn.save_parameters(fname + '.h5')
        print '--------------------------------------------------'
        print nn.get_parameters()
        print '--------------------------------------------------'

    def importNetwork(self, fname):
        print("Import Q-network from {}".format(fname))
        nn.load_parameters(fname + '.h5')
        print "Updating target Q-network"
        self.update_Q_target()
        print '--------------------------------------------------'
        print nn.get_parameters()
        print '--------------------------------------------------'

    def plotQ(self,clear=False):
        # Parameters
        grid_on = True
        v_max = 10. #np.max(self.Q[0, :, :])
        v_min = -50.
        x_labels = ["%.2f" % x for x in self.positions ]
        y_labels = ["%.2f" % y for y in self.velocities]
        titles = "Actions " + u"\u25C0" + ":push_left/" + u"\u25AA" + ":no_push/" + u"\u25B6" + ":push_right"
        Q = np.zeros((self.N_velocity*len(self.actions),self.N_position*len(self.actions)))
        for s_2 in range(len(self.velocities)):
            for s_1 in range(len(self.positions)):
                self.Q_target_x.d = [self.positions[s_1], self.velocities[s_2]]
                self.Q_target_Network.forward(clear_buffer=True)
                Q_hut = self.Q_target_Network.d.copy()
                #print "Q_x:" , self.Q_x.shape , "self.Q.d:", self.Q.d.shape
                for a in range(len(self.actions)):
                    Q[3 * s_2 + a, 3 * s_1 + 0] = Q_hut[0][0]
                    Q[3 * s_2 + a, 3 * s_1 + 1] = Q_hut[0][1]
                    Q[3 * s_2 + a, 3 * s_1 + 2] = Q_hut[0][2]
        im = axs.imshow(Q, interpolation='nearest', vmax=v_max, vmin=v_min, cmap=cm.jet)
        axs.grid(grid_on)
        axs.set_title(titles)
        axs.set_xlabel('Position')
        axs.set_ylabel('Velocity')
        x_start, x_end = axs.get_xlim()
        #y_start, y_end = axs.get_ylim()
        axs.set_xticks(np.arange(x_start, x_end, 3))
        axs.set_yticks(np.arange(x_start, x_end, 3))
        axs.set_xticklabels(x_labels, minor=False, fontsize='small', horizontalalignment='left', rotation=90)
        axs.set_yticklabels(y_labels, minor=False, fontsize='small', verticalalignment='top')
        self.cb = fig.colorbar(im, ax=axs)
        #
        plt.show(block=False)

    def plotQupdate(self):
        Q = np.zeros((self.N_velocity * len(self.actions), self.N_position * len(self.actions)))
        for s_2 in range(len(self.velocities)):
            for s_1 in range(len(self.positions)):
                self.Q_target_x.d = [self.positions[s_1], self.velocities[s_2]]
                self.Q_target_Network.forward(clear_buffer=True)
                Q_hut = self.Q_target_Network.d.copy()
                #print "Q_x:" , self.Q_x.shape , "self.Q.d:", self.Q.d.shape
                for a in range(len(self.actions)):
                    Q[3 * s_2 + a, 3 * s_1 + 0] = Q_hut[0][0]
                    Q[3 * s_2 + a, 3 * s_1 + 1] = Q_hut[0][1]
                    Q[3 * s_2 + a, 3 * s_1 + 2] = Q_hut[0][2]
        axs.get_images()[0].set_data(Q)
        axs.draw_artist(axs.images[0])
        fig.canvas.blit(axs.bbox)

    def plotLoss(self, step, error):
        if self.plot_reset:
            self.plot_reset = False
            if len(axs.lines) > 0:
                axs.lines[0].remove()
            axs.plot(step, error, color='blue', linestyle='-')
        else:
            x,y = axs.lines[0].get_data()
            x = np.append(x, step)
            y = np.append(y, error)
            axs.lines[0].set_data(x, y)
            axs.set_xlim(0, step)
            axs.set_ylim(min(y), max(y))
            #fig.canvas.draw()
            axs.draw_artist(axs.lines[0])
            fig.canvas.blit(axs.bbox)

    def clearTrajectory(self):
        self.plot_reset = True
        if len(axs.lines) > 0:
            axs.lines[0].remove()
        fig.canvas.draw()

    def plotTrajectory(self, state, action):
        pos = 1 + 3 * np.argmin(abs(self.positions - state[0]), axis=0)
        vel = 1 + 3 * np.argmin(abs(self.velocities - state[1]), axis=0)
        if self.plot_reset:
            self.plot_reset = False
            if len(axs.lines) > 0:
                axs.lines[0].remove()
            axs.plot(pos, vel, color='blue', linestyle='-')
        else:
            x, y = axs.lines[0].get_data()
            if x[-1] != pos or y[-1] != vel:
                x = np.append(x, pos)
                y = np.append(y, vel)
                axs.lines[0].set_data(x, y)
                axs.draw_artist(axs.lines[0])
                fig.canvas.blit(axs.bbox)

    def plotQaction(self):
        for i, vel in enumerate(self.velocities):
            for j, pos in enumerate(self.positions):
                self.Q_target_x.d = [pos, vel]
                self.Q_target_Network.forward(clear_buffer=True)
                Q = self.Q_target_Network.d.copy()
                action = np.argmax(Q[0])
                if action == 0:
                    axs.text(3 * j + .4, 3 * i + 1.6, u"\u25C0", fontsize=5)
                elif action == 1:
                    axs.text(3 * j + .4, 3 * i + 1.6, u"\u25AA", fontsize=5)
                else:
                    axs.text(3 * j + .4, 3 * i + 1.6, u"\u25B6", fontsize=5)
        fig.canvas.draw()

    # END NeuralQLearner class
