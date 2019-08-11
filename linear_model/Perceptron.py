import matplotlib.pyplot as plt
import numpy as np
from linear_model.LinearModel import LinearModel


class Perceptron(LinearModel):
    """Perceptron, optimized with SGD.

        Args:
        ------
        learning_rate: float, learning rate.
        train_step: int, number of training steps.
        rand_var: float, variance of normal distribution for weight initialization.
        rand_seed: int, random seed for random initialization.
        silent: boolean, whether to print training information in training process.
        """

    def __init__(self, learning_rate=0.001, train_step=10000,
                 rand_var=0.01, rand_seed=0, silent=False):
        batch_size = 1
        alpha = 0
        super(Perceptron, self).__init__(learning_rate, batch_size, train_step, alpha,
                                         rand_var, rand_seed, silent)

    def _get_gradients(self, x, y):
        """gradients calculation using one example."""
        dist = np.squeeze((np.dot(x, self.weight) + self.bias) * y)
        if dist > 0:
            loss = 0
            gradients_weight = np.zeros_like(self.weight)
            gradients_bias = 0
            gradients = (gradients_weight, gradients_bias)
            return gradients, loss
        else:
            loss = -dist
            gradients_weight = np.squeeze(-x * y, axis=0)
            gradients_bias = -y
            gradients = (gradients_weight, gradients_bias)
            return gradients, loss

    def _apply_gradients(self, g):
        """update weight using gradients."""
        self.weight -= self.lr * g[0]
        self.bias -= self.lr * g[1]

    def predict(self, x):
        dist = np.dot(x, self.weight) + self.bias
        return np.where(dist >= 0, 1, -1)

    def plot(self, x, y, num=10):
        n, m = x.shape
        assert n >= num, "Not enough samples to plot, {} is needed, only find {}.".format(num, n)
        assert m == 2, "Dimension of `x` is {}, but 2 is needed, only support 2-D plot.".format(m)
        plt.figure()
        idx = np.random.randint(0, n, num)
        x_plot, y_plot = x[idx], y[idx]
        idx_pos = np.argwhere(y_plot == 1)
        idx_neg = np.argwhere(y_plot == -1)

        plt.scatter(x_plot[idx_pos, 0], x_plot[idx_pos, 1], marker="o")
        plt.scatter(x_plot[idx_neg, 0], x_plot[idx_neg, 1], marker="^")
        x_min, x_max = min(x_plot[:, 0]), max(x_plot[:, 0])
        x_line = np.array([x_min, x_max])
        y_line = (-self.bias - self.weight[0] * x_line) / self.weight[1]
        plt.plot(x_line, y_line)
        plt.show()
