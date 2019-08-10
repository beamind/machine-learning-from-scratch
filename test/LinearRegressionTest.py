import logging
import numpy as np
from dataset.create_toy_data import toy_data_for_linear_regression
from utils.preprocessing import standardize
from linear_model.LinearRegression import LinearRegression

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    weight = np.array([2.0])
    bias = -1.0
    test = np.array([[0], [1], [2]])
    x, y = toy_data_for_linear_regression(weight=weight, bias=bias, n=1000, variance=0.5)
    model = LinearRegression(learning_rate=0.01,
                             batch_size=50,
                             train_step=5000,
                             alpha=0.0000,
                             rand_var=0.01,
                             rand_seed=0,
                             silent=False)
    model.fit(x, y)
    print("weight true/prediction:{}/{}".format(weight, model.weight))
    print("bias true/prediction:{}/{}".format(bias, model.bias))
    print(model.predict(test))
    model.plot(x, y, 20)
