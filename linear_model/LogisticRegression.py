import numpy as np
import matplotlib.pyplot as plt
from linear_model.LinearModel import LinearModel
from utils.function import sigmoid


class LogisticRegression(LinearModel):
    """Logistic regression with L2 regularization, optimized with SGD.

    Args:
    ------
    learning_rate: float, learning rate.
    batch_size: int, batch size.
    train_step: int, number of training steps.
    alpha: float, weight decay coefficient for L2 regularization. Must be positive or zero.
    rand_var: float, variance of normal distribution for weight initialization.
    rand_seed: int, random seed for random initialization.
    silent: boolean, whether to print training information in training process.
    """

    def __init__(self, learning_rate=0.001, batch_size=10, train_step=10000,
                 alpha=0.0, rand_var=0.01, rand_seed=0, silent=False):
        super(LogisticRegression, self).__init__(learning_rate, batch_size, train_step,
                                                 alpha, rand_var, rand_seed, silent)

    def _get_gradients(self, x, y):
        """gradients calculation using batch examples."""
        logit = np.dot(x, self.weight) + self.bias
        prob = sigmoid(logit)
        loss = np.mean(-y * np.log(prob) - (1 - y) * np.log(1 - prob)) + 0.5 * self.alpha * np.square(self.weight).sum()
        error = np.expand_dims(prob - y, axis=1)
        gradients_weight = (error * x).mean(axis=0) + self.alpha * self.weight
        gradients_bias = np.mean(error)
        gradients = (gradients_weight, gradients_bias)
        return gradients, loss

    def _apply_gradients(self, g):
        """update weight using gradients."""
        self.weight -= self.lr * g[0]
        self.bias -= self.lr * g[1]

    def predict(self, x, threshold=0.5):
        logit = np.dot(x, self.weight) + self.bias
        prob = sigmoid(logit)
        labels = np.where(prob > threshold, 1, 0)
        return prob, labels

    def plot(self, x, y, num=10):
        n, m = x.shape
        assert n >= num, "Not enough samples to plot, {} is needed, only find {}.".format(num, n)
        assert m == 2, "Dimension of `x` is {}, but 2 is needed, only support 2-D plot.".format(m)
        plt.figure()
        idx = np.random.randint(0, n, num)
        x_plot, y_plot = x[idx], y[idx]
        idx_pos = np.argwhere(y_plot == 1)
        idx_neg = np.argwhere(y_plot == 0)

        plt.scatter(x_plot[idx_pos, 0], x_plot[idx_pos, 1], marker="o")
        plt.scatter(x_plot[idx_neg, 0], x_plot[idx_neg, 1], marker="^")
        x_min, x_max = min(x_plot[:, 0]), max(x_plot[:, 0])
        x_line = np.array([x_min, x_max])
        y_line = (-self.bias - self.weight[0] * x_line) / self.weight[1]
        plt.plot(x_line, y_line)
        plt.show()
