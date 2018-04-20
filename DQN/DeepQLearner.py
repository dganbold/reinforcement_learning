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
from nnabla.monitor import tile_images

from args import get_args

import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#
#
#isNatureVersion = True
isNatureVersion = False # NIPS version of DQN
#
#
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

    def getsize(self):
        return len(self.memory)

    def clear(self):
        del self.memory[:]
# END Experience class
#
#
class Q_Network:
    def __init__(self, n_states, n_actions, parameters, number_of_layers, batch_size=int(128)):
        self.Q_x = nn.Variable([batch_size, n_states[2], n_states[0], n_states[1]])
        self.parameter_list = list()
        for n in range(number_of_layers):
            self.parameter_list.append(nn.Variable(parameters[2*n+0][1].d.shape, need_grad=False))  # Weights
            self.parameter_list.append(nn.Variable(parameters[2*n+1][1].d.shape, need_grad=False))  # Biases
        self.reflect(parameters)
        #
        if isNatureVersion: # Nature vesion
            c1 = F.relu(F.convolution(self.Q_x, self.parameter_list[0], self.parameter_list[1], pad=(0, 0), stride=(4, 4)), inplace=True)
            c2 = F.relu(F.convolution(c1, self.parameter_list[2], self.parameter_list[3], pad=(0, 0), stride=(2, 2)), inplace=True)
            c3 = F.relu(F.affine(c2, self.parameter_list[4], self.parameter_list[5]), inplace=True)
            self.Q_Network = F.affine(c3, self.parameter_list[6], self.parameter_list[7])
        else:               # NIPS version
            c1 = F.relu(F.convolution(self.Q_x, self.parameter_list[0], self.parameter_list[1], pad=(0, 0), stride=(4, 4)), inplace=True)
            c2 = F.relu(F.convolution(c1, self.parameter_list[2], self.parameter_list[3], pad=(0, 0), stride=(2, 2)), inplace=True)
            c3 = F.relu(F.convolution(c2, self.parameter_list[4], self.parameter_list[5], pad=(0, 0), stride=(1, 1)), inplace=True)
            c4 = F.relu(F.affine(c3, self.parameter_list[6], self.parameter_list[7]), inplace=True)
            self.Q_Network = F.affine(c4, self.parameter_list[8], self.parameter_list[9])
    #
    #
    def reflect(self, parameters):
        for n in range(len(self.parameter_list)):
            self.parameter_list[n].d = parameters[n][1].d.copy()
            #print("parameters.shape:{}".format(self.parameter_list[n].d.shape))
    #
    #
    def forward(self, input):       # Perform the forward calculation on single input
        for frame in range(4):
            self.Q_x.d[0,frame,:,:] = input[:,:,frame]
        self.Q_Network.forward(clear_buffer=True)
        return self.Q_Network.d.copy()[0]
    #
    #
    def batch_forward(self, input): # Perform the forward calculation on batch input
        self.Q_x.d = input  # batch input
        self.Q_Network.forward(clear_buffer=True)
        return self.Q_Network.d.copy()
#
#
class DeepQLearner:
    def __init__(self, n_states, actions, batch_size=int(32), update_target=1000, epsilon=0.1, gamma=0.99, learning_rate=1e-3):
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
        self.learning_rate = learning_rate
        #self.learning_rate = 0.00025 #1e-3
        self.gradient_momentum = 0.95
        self.squared_gradient_momentum = 0.95
        self.batch_size = batch_size
        self.model_save_path = 'models'
        #self.model_save_interval = 1000
        self.weight_decay = args.weight_decay
        # --------------------------------------------------
        print "Initializing the Neural Network."
        # --------------------------------------------------
        # Preparing the Computation Graph for Neural fitted Q function
        self.Q_x = nn.Variable([self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]])
        self.Q_y = nn.Variable([self.batch_size, self.n_actions])
        # Construct DeepNetwork for Q-Learning
        if isNatureVersion: # Nature vesion
            self.number_of_layers = 4
            self.hidden_neurons = 256    # Hidden layer's neuron number
            c1 = F.relu(PF.convolution(self.Q_x, 32, (8, 8), pad=(0, 0), stride=(4, 4), name='conv1'), inplace=True)
            c2 = F.relu(PF.convolution(c1, 32, (4, 4), pad=(0, 0), stride=(2, 2), name='conv2'), inplace=True)
            c3 = F.relu(PF.affine(c2, self.hidden_neurons, name='fc3'), inplace=True)
            self.Q_Network = PF.affine(c3, self.n_actions, name='fc4')
            print("-- Nature DQN architecture -------------")
            print("input.shape:{}".format(self.Q_x.shape))
            print("conv1.shape:{}".format(c1.d.shape))
            print("conv2.shape:{}".format(c2.d.shape))
            print("conv3.shape:{}".format(c3.d.shape))
            print("  fc4.shape:{}".format(self.Q_Network.d.shape))
            print("----------------------------------------")
        else:               # NIPS version
            self.number_of_layers = 5
            self.hidden_neurons = 512    # Hidden layer's neuron number
            c1 = F.relu(PF.convolution(self.Q_x, 32, (8, 8), pad=(0, 0), stride=(4, 4), name='conv1'), inplace=True)
            c2 = F.relu(PF.convolution(c1, 32, (4, 4), pad=(0, 0), stride=(2, 2), name='conv2'), inplace=True)
            c3 = F.relu(PF.convolution(c2, 64, (3, 3), pad=(0, 0), stride=(1, 1), name='conv3'), inplace=True)
            c4 = F.relu(PF.affine(c3, self.hidden_neurons, name='fc4'), inplace=True)
            self.Q_Network = PF.affine(c4, self.n_actions, name='fc5')
            print("-- NIPS DQN architecture ---------------")
            print("input.shape:{}".format(self.Q_x.shape))
            print("conv1.shape:{}".format(c1.d.shape))
            print("conv2.shape:{}".format(c2.d.shape))
            print("conv3.shape:{}".format(c3.d.shape))
            print("  fc4.shape:{}".format(c4.d.shape))
            print("  fc5.shape:{}".format(self.Q_Network.d.shape))
            print("----------------------------------------")
        #
        self.Q_Network.persistent = True
        self.Q_Network_parameters = nn.get_parameters().items()
        # Create loss function.
        #self.loss = F.mean(F.squared_error(self.Q_Network, self.Q_y))
        self.loss = F.mean(F.huber_loss(self.Q_Network, self.Q_y))
        # Preparing the Computation Graph for target Q-Network
        # Construct target Q-Network for Q-Learning.
        self.Q_target_Network = self.clone_Q_network()
        # --------------------------------------------------
        print "Initializing the Solver."
        # --------------------------------------------------
        # Create Solver
        #self.solver = S.Sgd(self.learning_rate)
        self.solver = S.Adam(self.learning_rate)
        #self.solver = S.RMSprop(self.learning_rate, self.gradient_momentum)
        self.solver.set_parameters(nn.get_parameters())
        self.update_Q_target = update_target
        self.iter = 0
    #
    #
    def train_Q_network(self, input):
        # Perform a minibatch Q-learning update
        # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # ----------------------------------------------
        # Preparing data
        s0_vector = np.zeros((self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]))
        s1_vector = np.zeros((self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]))
        for n in range(self.batch_size):
            for frame in range(4):
                s0_vector[n,frame,:,:] = input[n][0][0][:,:,frame]  # state_t
                s1_vector[n,frame,:,:] = input[n][0][3][:,:,frame]  # state_t+1
        # ----------------------------------------------
        # Prediction of Q-Value on current state.
        self.Q_x.d = s0_vector.copy()
        self.Q_Network.forward(clear_buffer=True)
        Q_present = self.Q_Network.d.copy()
        # ----------------------------------------------
        # Prediction of Q-Value at next state by Target Deep Q-Network.
        Q_next = self.Q_target_Network.batch_forward(s1_vector.copy())
        maxQ = np.amax(Q_next, axis=1)
        # ----------------------------------------------
        # Calculate target value of Q-network
        target = Q_present.copy()
        for n in range(self.batch_size):
            a = np.argmin(abs(self.actions - input[n][0][1] ), axis=0)
            if input[n][1]:     # terminal state
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
        #print self.Q_Network_parameters[3][1].d[1]
        # Every C updates clone the Q-network parameters to target Q-network
        if self.iter % self.update_Q_target == 0:
            #print "Updating target Q-network"
            self.Q_target_Network.reflect(self.Q_Network_parameters)
    #
    #
    def clone_Q_network(self):
        return Q_Network(self.n_states,self.n_actions,self.Q_Network_parameters,self.number_of_layers,self.batch_size)
    #
    #
    def training_error(self, y_pred, y_taget):
        mse = ((y_pred - y_taget) ** 2).sum(axis=1).mean()
        return mse
    #
    #
    def eGreedy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.greedy(state)
        return action
    #
    #
    def greedy(self, state):
        Q = self.Q_target_Network.forward(state)
        maxQ = np.max(Q)
        best = [i for i in range(len(self.actions)) if Q[i] == maxQ]
        if len(best) > 1:
            i = random.choice(best)
        else:
            i = np.argmax(Q)
        return self.actions[i]
    #
    #
    def exportNetwork(self, fname):
        print("Export Q-network to {}".format(fname))
        nn.save_parameters(fname + '.h5')
        #print '--------------------------------------------------'
        #print nn.get_parameters()
        #print '--------------------------------------------------'
    #
    #
    def importNetwork(self, fname):
        print("Import Q-network from {}".format(fname))
        nn.load_parameters(fname + '.h5')
        print "Updating target Q-network"
        self.Q_target_Network.reflect(self.Q_Network_parameters)
        #print '--------------------------------------------------'
        #print nn.get_parameters()
        #print '--------------------------------------------------'

    # END DeepQLearner class
