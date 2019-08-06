import math
import numpy as np


class LinearRegression(object):
    """Linear least square with L2 regularization, optimized with SGD.

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

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=10,
                 train_step=10000,
                 alpha=0.0,
                 rand_var=0.01,
                 rand_seed=0,
                 silent=False):
        self.lr = learning_rate
        self.bsz = batch_size
        self.train_step = train_step
        self.alpha = alpha
        self.rand_var = rand_var
        self.rand_seed = rand_seed
        self.silent = silent
        self.weight = None
        self.loss = 0

    def fit(self, x, y):
        """
        Args:
        ------
        x: features of training data, assuming the first column
           is all 1. shape=[n_examples, n_features].
        y: targets of training data. shape=[n_examples].
        """

        n, m = x.shape
        np.random.seed(self.rand_seed)
        self.weight = np.random.randn(m) * math.sqrt(self.rand_var)
        for i in range(self.train_step):
            batch_index = np.random.randint(0, n, self.bsz)
            x_batch, y_batch = x[batch_index], y[batch_index]
            gradients, self.loss = self._get_gradients(x_batch, y_batch)
            self._apply_gradients(gradients)
            if not self.silent and i % 100 == 0:
                print("step {}: ,loss: {:.5f}".format(i, self.loss))

    def _get_gradients(self, x, y):
        """gradients calculation using batch examples."""
        y_pred = np.dot(x, self.weight)
        error = np.expand_dims(y_pred - y, axis=1)
        loss = np.square(error).mean()
        if self.alpha > 0.0:
            loss += self.alpha * np.square(self.weight[1:]).sum()
        gradients = (error * x).mean(axis=0)
        if self.alpha > 0.0:
            # L2 regularization don't penalize bias.
            gradients += self.alpha * np.concatenate([[0], self.weight[1:]], axis=0)
        return gradients, loss

    def _apply_gradients(self, g):
        """update weight using gradients."""
        self.weight -= self.lr * g
