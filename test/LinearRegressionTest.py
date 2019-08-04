import numpy as np
from dataset.create_toy_data import toy_data_for_linear_regression
from linear_model import LinearRegression

if __name__ == "__main__":
    weight = np.array([-0.01, 0.1, -1.0, 10, -100])
    x, y = toy_data_for_linear_regression(weight=weight, n=10000, m=5)
    model = LinearRegression.LinearRegression(learning_rate=0.01,
                                              batch_size=10,
                                              train_step=10000,
                                              rand_var=0.0001,
                                              rand_seed=0,
                                              silent=False)
    model.fit(x, y)
    print(model.weight)
