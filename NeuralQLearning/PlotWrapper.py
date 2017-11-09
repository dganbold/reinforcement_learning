
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class PlotTimeSeries(object):

    def __init__(self, x_lim=[0,1], y_lim=[0,1], size=(5.0, 5.0)):
        self.x_lim = x_lim
        self.y_lim = y_lim
        #
        fig, axs = plt.subplots(1, 1, figsize=size)
        self.fig = fig
        self.axs = axs
        self.grid_on = False
        self.plot_reset = False
        #
        self.alpha = 0.0799

    def create(self, data, x_label, y_label, title):
        # Parameters
        self.axs.plot(data[0], data[1], label='original', color='blue', linestyle='-', alpha=0.3, linewidth=2.0)
        self.axs.plot(data[0], data[1], label='smoothed', color='red',  linestyle='-', alpha=1.0, linewidth=1.0)
        self.axs.set_title(title)
        self.axs.set_xlabel(x_label)
        self.axs.set_ylabel(y_label)
        self.axs.set_xlim(self.x_lim[0], self.x_lim[1])
        self.axs.set_ylim(self.y_lim[0], self.y_lim[1])
        self.axs.set_axis_bgcolor('#dcdcdc')
        legend = self.axs.legend(loc='upper right', shadow=True)
        # Draw
        plt.show(block=False)

    def append(self, data):
        x, y = self.axs.lines[0].get_data()
        x = np.append(x, data[0])
        y = np.append(y, data[1])
        self.axs.lines[0].set_data(x, y)
        # self.axs.set_ylim(min(y), max(y))
        if self.x_lim[1] < data[0]:
            self.axs.set_xlim(self.x_lim[0], data[0])
            # Smoothing
            x, y = self.axs.lines[1].get_data()
            y_smoot = y[-1] + self.alpha*(data[1] - y[-1])
            x = np.append(x, data[0])
            y = np.append(y, y_smoot)
            self.axs.lines[1].set_data(x, y)
            # Draw
            self.fig.canvas.draw()
        else:
            self.axs.lines[1].set_data(x, y)
            # Draw
            self.axs.draw_artist(self.axs.lines[0])
            self.fig.canvas.blit(self.axs.bbox)

# END PlotWrapper class


class Plot2D(object):

    def __init__(self, x_lim=[0,1], y_lim=[0,1], x_n=1, y_n=1, size=(5.0, 5.0)):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_n = x_n
        self.y_n = y_n
        self.x_grid = np.linspace(self.x_lim[0], self.x_lim[1], num=self.x_n, endpoint=True)
        self.y_grid = np.linspace(self.y_lim[0], self.y_lim[1], num=self.y_n, endpoint=True)
        #
        fig, axs = plt.subplots(1, 1, figsize=size)
        self.fig = fig
        self.axs = axs
        self.grid_on = False
        self.plot_reset = False

    def create(self, data, x_label, y_label, title):
        # Parameters
        v_max = 20.  # np.max(self.Q[0, :, :])
        v_min = -50.
        x_labels = ["%.3f" % x for x in self.x_grid]
        y_labels = ["%.3f" % y for y in self.y_grid]
        #
        data = np.reshape(data, (self.y_n, self.x_n))
        im = self.axs.imshow(data, interpolation='nearest', vmax=v_max, vmin=v_min, cmap=cm.jet)
        self.axs.grid(self.grid_on)
        self.axs.set_title(title)
        self.axs.set_xlabel(x_label)
        self.axs.set_ylabel(y_label)
        x_start, x_end = self.axs.get_xlim()
        # y_start, y_end = axs.get_ylim()
        self.axs.set_xticks(np.arange(x_start, x_end, 1))
        self.axs.set_yticks(np.arange(x_start, x_end, 1))
        self.axs.set_xticklabels(x_labels, minor=False, fontsize='small', horizontalalignment='left', rotation=90)
        self.axs.set_yticklabels(y_labels, minor=False, fontsize='small', verticalalignment='top')
        self.cb = self.fig.colorbar(im, ax=self.axs)
        # Draw
        plt.show(block=False)

    def update(self, data):
        data = np.reshape(data, (self.y_n, self.x_n))
        self.axs.get_images()[0].set_data(data)
        # Draw
        self.axs.draw_artist(self.axs.images[0])
        self.fig.canvas.blit(self.axs.bbox)

    def appendPoint(self, data):
        x = np.argmin(abs(self.x_grid - data[0]), axis=0)
        y = np.argmin(abs(self.y_grid - data[1]), axis=0)
        if len(self.axs.lines) == 0:
            self.axs.plot(x, y, color='black', linestyle='-')
            # Draw
            plt.show(block=False)
        else:
            x_list, y_list = self.axs.lines[0].get_data()
            if x_list[-1] != x or y_list[-1] != y:
                x_list = np.append(x_list, x)
                y_list = np.append(y_list, y)
                self.axs.lines[0].set_data(x_list, y_list)
                self.axs.draw_artist(self.axs.lines[0])
                self.fig.canvas.blit(self.axs.bbox)

    def deletePoints(self):
        if len(self.axs.lines) > 0:
            self.axs.lines[0].remove()
            # Draw
            plt.show(block=False)

    def getMeshGrid(self):
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        return zip(np.ravel(X), np.ravel(Y))

# END Plot2D class
