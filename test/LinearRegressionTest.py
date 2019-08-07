import numpy as np
from dataset.create_toy_data import toy_data_for_linear_regression
from linear_model import LinearRegression

if __name__ == "__main__":
    weight = np.array([-0.01, 0.1, -1.0, 10, -100])
    bias = 1
    test = np.array([[100, 10, 1, 0.1, -1], [-100, 10, 1, 10, 1]])
    x, y = toy_data_for_linear_regression(weight=weight, bias=bias, n=10000, variance=0.01)
    model = LinearRegression.LinearRegression(learning_rate=0.001,
                                              batch_size=100,
                                              train_step=10000,
                                              alpha=0.0000,
                                              rand_var=0.0001,
                                              rand_seed=0,
                                              silent=False)
    model.fit(x, y)
    print(model.weight)
    print(model.bias)
    print(model.predict(test))
