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


class Experience(object):

    def __init__(self, max_memory=1000):
        self.max_memory = max_memory
        self.memory = list()

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

    def clear(self):
        del self.memory[:]

    # END Experience class


class NeuralQLearner:
    def __init__(self, n_states, actions, hidden_neurons=100, batch_size=int(128), update_target=100, epsilon=0.1, gamma=0.9):
        # Get context.
        from nnabla.contrib.context import extension_context
        args = get_args()
        # print "weight_decay:", args.weight_decay
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        logger.info("Running in %s" % extension_module)
        ctx = extension_context(extension_module, device_id=args.device_id)
        nn.set_default_context(ctx)

        # Q-Learing parametes
        self.epsilon = epsilon
        self.gamma = gamma

        self.actions = actions
        self.n_actions = len(actions)
        self.n_states = n_states

        # Neural network's training parametes
        self.learning_rate = 1e-3
        self.gradient_momentum = 0.95
        self.squared_gradient_momentum = 0.95
        # Hidden layer's neuron number
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.model_save_path = 'models'
        self.model_save_interval = 1000
        self.weight_decay = args.weight_decay

        # --------------------------------------------------
        print "Initializing the Neural Network."
        # --------------------------------------------------
        # Preparing the Computation Graph for Q
        self.Q_x = nn.Variable([self.batch_size, self.n_states])
        self.Q_y = nn.Variable([self.batch_size, self.n_actions])

        # Construct Q-Network for Q-Learning.
        l1 = F.tanh(PF.affine(self.Q_x, self.hidden_neurons, name='affine1'))
        # l1 = F.relu(PF.affine(self.Q_x, self.hidden_neurons, name='affine1'))
        self.Q_Network = PF.affine(l1, self.n_actions, name='affine2')
        self.Q_Network.persistent = True

        # Create loss function.
        # self.loss = F.mean(F.squared_error(self.Q_Network, self.Q_y))
        self.loss = F.mean(F.huber_loss(self.Q_Network, self.Q_y))

        # Preparing the Computation Graph for target Q-Network
        self.Q_target_x = nn.Variable([self.batch_size, self.n_states])
        self.Q_target_w1 = nn.Variable([self.n_states, self.hidden_neurons], need_grad=False)   # Weights
        self.Q_target_b1 = nn.Variable([self.hidden_neurons], need_grad=False)                  # Biases
        self.Q_target_w2 = nn.Variable([self.hidden_neurons, self.n_actions], need_grad=False)  # Weights
        self.Q_target_b2 = nn.Variable([self.n_actions], need_grad=False)      # Biases

        # Construct target Q-Network for Q-Learning.
        h1 = F.tanh(F.affine(self.Q_target_x, self.Q_target_w1, self.Q_target_b1))
        # h1 = F.relu(F.affine(self.Q_target_x, self.Q_target_w1, self.Q_target_b1))
        self.Q_target_Network = F.affine(h1, self.Q_target_w2, self.Q_target_b2)
        self.update_Q_target()

        # --------------------------------------------------
        print "Initializing the Solver."
        # --------------------------------------------------
        # Create Solver
        #self.solver = S.Sgd(self.learning_rate)
        self.solver = S.RMSprop(self.learning_rate, self.gradient_momentum)
        self.solver.set_parameters(nn.get_parameters())
        self.update_Q = update_target
        self.iter = 0
        #

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

    def evaluate_Q_target(self, input):
        result = list()
        for n in range(len(input)):
            self.Q_target_x.d = input[n]
            self.Q_target_Network.forward(clear_buffer=True)
            Q_hut = self.Q_target_Network.d.copy()
            result.append(max(Q_hut[0]))
        return result

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

    # END NeuralQLearner class
