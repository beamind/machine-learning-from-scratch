import numpy as np
from dataset.create_toy_data import toy_data_for_logistic_regression
from linear_model.LogisticRegression import LogisticRegression

if __name__ == "__main__":
    weight = np.array([-1.0, 2.0])
    bias = 3.0
    test = np.array([[0, 0], [1, 1], [2, 2]])
    x, y = toy_data_for_logistic_regression(weight=weight, bias=bias, n=1000, variance=0.05)
    model = LogisticRegression(learning_rate=0.3,
                               batch_size=100,
                               train_step=5000,
                               alpha=0.03,
                               rand_var=0.01,
                               rand_seed=0,
                               silent=False)
    model.fit(x, y)
    print("weight true/prediction:{}/{}".format(weight, model.weight))
    print("bias true/prediction:{}/{}".format(bias, model.bias))
    print(model.predict(test))
    model.plot(x, y, 100)
