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
isNatureVersion = True
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
    def __init__(self, n_states, n_actions, parameters, hidden_neurons=256, batch_size=int(128)):
        self.Q_x  = nn.Variable([batch_size, n_states[2], n_states[0], n_states[1]])
        self.c1_w = nn.Variable(parameters[0][0].shape, need_grad=False)   # Weights
        self.c1_b = nn.Variable(parameters[0][1].shape, need_grad=False)   # Biases
        self.c2_w = nn.Variable(parameters[1][0].shape, need_grad=False)   # Weights
        self.c2_b = nn.Variable(parameters[1][1].shape, need_grad=False)   # Biases
        self.c3_w = nn.Variable(parameters[2][0].shape, need_grad=False)   # Weights
        self.c3_b = nn.Variable(parameters[2][1].shape, need_grad=False)   # Biases
        self.c4_w = nn.Variable(parameters[3][0].shape, need_grad=False)   # Weights
        self.c4_b = nn.Variable(parameters[3][1].shape, need_grad=False)   # Biases
        #self.c5_w = nn.Variable(parameters[4][0].shape, need_grad=False)   # Weights
        #self.c5_b = nn.Variable(parameters[4][1].shape, need_grad=False)   # Biases
        #
        self.reflect(parameters)
        #
        if isNatureVersion:
            c1 = F.relu(F.convolution(self.Q_x, self.c1_w, self.c1_b, pad=(0, 0), stride=(4, 4)), inplace=True)
            c2 = F.relu(F.convolution(c1, self.c2_w, self.c2_b, pad=(0, 0), stride=(2, 2)), inplace=True)
            c3 = F.relu(F.affine(c2, self.c3_w, self.c3_b), inplace=True)
            self.Q_Network = F.affine(c3, self.c4_w, self.c4_b)
        else:
            c1 = F.relu(F.convolution(self.Q_x, self.c1_w, self.c1_b, pad=(0, 0), stride=(4, 4)), inplace=True)
            c2 = F.relu(F.convolution(c1, self.c2_w, self.c2_b, pad=(0, 0), stride=(2, 2)), inplace=True)
            c3 = F.relu(F.convolution(c2, self.c3_w, self.c3_b, pad=(0, 0), stride=(1, 1)), inplace=True)
            c4 = F.relu(F.affine(c3, self.c4_w, self.c4_b), inplace=True)
            self.Q_Network = F.affine(c4, self.c5_w, self.c5_b)
    #
    #
    def reflect(self, parameters):
        self.c1_w.d = parameters[0][0].copy()
        self.c1_b.d = parameters[0][1].copy()
        self.c2_w.d = parameters[1][0].copy()
        self.c2_b.d = parameters[1][1].copy()
        self.c3_w.d = parameters[2][0].copy()
        self.c3_b.d = parameters[2][1].copy()
        self.c4_w.d = parameters[3][0].copy()
        self.c4_b.d = parameters[3][1].copy()
        #self.c5_w.d = parameters[4][0].copy()
        #self.c5_b.d = parameters[4][1].copy()
    #
    #
    def forward(self, input):
        self.Q_x.d[0,0,:,:] = input[:,:,0]
        self.Q_x.d[0,1,:,:] = input[:,:,1]
        self.Q_x.d[0,2,:,:] = input[:,:,2]
        self.Q_x.d[0,3,:,:] = input[:,:,3]
        self.Q_Network.forward(clear_buffer=True)
        return self.Q_Network.d.copy()[0]
    #
    #
    def batch_forward(self, input):
        self.Q_x.d = input  # bath input
        self.Q_Network.forward(clear_buffer=True)
        return self.Q_Network.d.copy()
#
#
class DeepQLearner:
    def __init__(self, n_states, actions, hidden_neurons=512, batch_size=int(32), update_target=1000, epsilon=0.1, gamma=0.99, learning_rate=1e-3):
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
        # Hidden layer's neuron number
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.model_save_path = 'models'
        #self.model_save_interval = 1000
        self.weight_decay = args.weight_decay

        # --------------------------------------------------
        print "Initializing the Neural Network."
        # --------------------------------------------------
        # Preparing the Computation Graph for Q
        #print self.n_states
        self.Q_x = nn.Variable([self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]])
        self.Q_y = nn.Variable([self.batch_size, self.n_actions])
        print("Q_x.shape:{}".format(self.Q_x.shape))
        print("Q_y.shape:{}".format(self.Q_y.shape))

        # Construct DeepNetwork for Q-Learning
        self.number_of_layers = 4
        c1 = F.relu(PF.convolution(self.Q_x, 32, (8, 8), pad=(0, 0), stride=(4, 4), name='conv1'), inplace=True)
        c2 = F.relu(PF.convolution(c1, 32, (4, 4), pad=(0, 0), stride=(2, 2), name='conv2'), inplace=True)
        #c3 = F.relu(PF.convolution(c2, 64, (3, 3), pad=(0, 0), stride=(1, 1), name='conv3'), inplace=True)
        c3 = F.relu(PF.affine(c2, self.hidden_neurons, name='fc3'), inplace=True)
        self.Q_Network = PF.affine(c3, self.n_actions, name='fc4')
        self.Q_Network.persistent = True
        self.Q_Network_parameters = nn.get_parameters().items()
        #
        print("c1.shape:{}".format(c1.d.shape))
        print("c2.shape:{}".format(c2.d.shape))
        print("c3.shape:{}".format(c3.d.shape))
        print("self.Q_Network.shape:{}".format(self.Q_Network.d.shape))
        #
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

    def train_Q_network(self, input):
        # Perform a minibatch Q-learning update
        # input[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # ----------------------------------------------
        # Preparing data
        s0_vector = np.zeros((self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]))
        s1_vector = np.zeros((self.batch_size, self.n_states[2], self.n_states[0], self.n_states[1]))
        for n in range(self.batch_size):
            s0_vector[n,0,:,:] = input[n][0][0][:,:,0]  # state_t
            s0_vector[n,1,:,:] = input[n][0][0][:,:,1]  # state_t
            s0_vector[n,2,:,:] = input[n][0][0][:,:,2]  # state_t
            s0_vector[n,3,:,:] = input[n][0][0][:,:,3]  # state_t
            s1_vector[n,0,:,:] = input[n][0][3][:,:,0]  # state_t+1
            s1_vector[n,1,:,:] = input[n][0][3][:,:,1]  # state_t+1
            s1_vector[n,2,:,:] = input[n][0][3][:,:,2]  # state_t+1
            s1_vector[n,3,:,:] = input[n][0][3][:,:,3]  # state_t+1
        # ----------------------------------------------
        # Prediction of Q-Value on current state.
        self.Q_x.d = s0_vector.copy()
        self.Q_Network.forward(clear_buffer=True)
        Q_present = self.Q_Network.d.copy()
        # ----------------------------------------------
        # Prediction of Q-Value at next state by Target Deep Q-Network.
        Q_next = self.Q_target_Network.batch_forward(s1_vector.copy())
        #print("Q_next.shape:{}".format(Q_next.shape))
        #Q_next = self.Q_Network.d.copy()
        maxQ = np.amax(Q_next, axis=1)
        #print("maxQ.shape:{}".format(maxQ.shape))
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
        #print self.Q_Network_parameters[3][1].d[1]
        # Every C updates clone the Q-network parameters to target Q-network
        if self.iter % self.update_Q_target == 0:
            #print "Updating target Q-network"
            parameters = self.clone_Q_network_parameters()
            self.Q_target_Network.reflect(parameters)
    #
    #
    def clone_Q_network(self):
        parameters = self.clone_Q_network_parameters()
        return Q_Network(self.n_states,self.n_actions,parameters,self.hidden_neurons,self.batch_size)
    #
    #
    def clone_Q_network_parameters(self):
        parameters = list()
        for n in range(self.number_of_layers):
            parameters.append([ self.Q_Network_parameters[2*n][1].d.copy(), self.Q_Network_parameters[2*n+1][1].d.copy() ])
        return parameters
    #
    #
    #def evaluate_Q_target(self, input):
    #    result = list()
    #    for n in range(len(input)):
    #        self.Q_target_x.d = input[n]
    #        self.Q_target_Network.forward(clear_buffer=True)
    #        Q_hut = self.Q_target_Network.d.copy()
    #        result.append(max(Q_hut[0]))
    #    return result
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
        print '--------------------------------------------------'
        print nn.get_parameters()
        print '--------------------------------------------------'
    #
    #
    def importNetwork(self, fname):
        print("Import Q-network from {}".format(fname))
        nn.load_parameters(fname + '.h5')
        print "Updating target Q-network"
        parameters = self.clone_Q_network_parameters()
        self.Q_target_Network.reflect(parameters)
        print '--------------------------------------------------'
        print nn.get_parameters()
        print '--------------------------------------------------'

    # END DeepQLearner class
