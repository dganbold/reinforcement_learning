#
#
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, axs = plt.subplots(1, 1, figsize=(5.8, 5))


class TabularQLearner:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        # Q-Table
        self.plim = [-1.2, 0.6]
        self.vlim = [-0.07, 0.07]
        self.N_position = 27
        self.N_velocity = 27
        self.positions = np.linspace(self.plim[0], self.plim[1], num=self.N_position, endpoint=True)
        self.velocities = np.linspace(self.vlim[0], self.vlim[1], num=self.N_velocity, endpoint=True)
        self.Q = (-1.)*np.ones((self.N_velocity,self.N_position*len(self.actions)))
        #
        self.plot_reset = True

    def getQ(self, state, action):
        pos = np.argmin(abs(self.positions - state[0]), axis=0)
        vel = np.argmin(abs(self.velocities - state[1]), axis=0)
        return self.Q[vel,3*pos + action]

    def learnQ(self, state1, action1, reward, state2):
        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        pos = np.argmin(abs(self.positions - state1[0]), axis=0)
        vel = np.argmin(abs(self.velocities - state1[1]), axis=0)
        Q_n = self.Q[vel,3*pos + action1]
        maxQ = max([self.getQ(state2, a) for a in self.actions])
        self.Q[vel, 3 * pos + action1] = Q_n + self.alpha * (reward + self.gamma*maxQ - Q_n)

    def eGreedy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.greedy(state)
        return action

    def greedy(self, state):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        best = [i for i in range(len(self.actions)) if q[i] == maxQ]
        # In case there're several state-action max values
        # policy select a random one of them
        if len(best) > 1:
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        return self.actions[i]

    def exportQ(self,fname):
        print '--------------------------------------------------'
        print self.Q
        print '--------------------------------------------------'
        print("Export Q-table to {}".format(fname))
        np.save(fname, self.Q)

    def importQ(self,fname):
        print("Import Q-table from {}".format(fname))
        self.Q = np.load(fname + '.npy')
        print '--------------------------------------------------'
        print self.Q
        print '--------------------------------------------------'

    def printQ(self):
        print '--------------------------------------------------'
        print self.Q

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
            Q[3 * s_2 + 0,:] = self.Q[s_2,:]
            Q[3 * s_2 + 1,:] = self.Q[s_2,:]
            Q[3 * s_2 + 2,:] = self.Q[s_2,:]
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
            Q[3 * s_2 + 0,:] = self.Q[s_2,:]
            Q[3 * s_2 + 1,:] = self.Q[s_2,:]
            Q[3 * s_2 + 2,:] = self.Q[s_2,:]
        axs.get_images()[0].set_data(Q)
        #fig.canvas.draw()
        axs.draw_artist(axs.images[0])
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
            x,y = axs.lines[0].get_data()
            if x[-1] != pos or y[-1] != vel:
                x = np.append(x, pos)
                y = np.append(y, vel)
                axs.lines[0].set_data(x, y)
                axs.draw_artist(axs.lines[0])
                fig.canvas.blit(axs.bbox)

    def plotQaction(self):
        for vel in range(len(self.velocities)):
            for pos in range(len(self.positions)):
                q = [self.Q[vel, 3 * pos + a] for a in self.actions]
                if max(q) != -float('inf'):
                    action = np.argmax(q)
                    if action == 0:
                        axs.text(3 * pos + .4, 3 * vel + 1.6, u"\u25C0", fontsize=5)
                    elif action == 1:
                        axs.text(3 * pos + .4, 3 * vel + 1.6, u"\u25AA", fontsize=5)
                    else:
                        axs.text(3 * pos + .4, 3 * vel + 1.6, u"\u25B6", fontsize=5)
        fig.canvas.draw()

    # END TabularQLearner class
