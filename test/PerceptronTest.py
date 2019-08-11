import numpy as np
from dataset.create_toy_data import toy_data_for_perceptron
from linear_model.Perceptron import Perceptron

if __name__ == "__main__":
    weight = np.array([-1.0, 2.0])
    bias = 3.0
    test = np.array([[0, 0], [1, 1], [2, 2]])
    x, y = toy_data_for_perceptron(weight=weight, bias=bias, n=1000, variance=0.05)
    model = Perceptron(learning_rate=0.01,
                       train_step=15000,
                       rand_var=0.01,
                       rand_seed=0,
                       silent=False)
    model.fit(x, y)
    print("weight true/prediction:{}/{}".format(weight, model.weight))
    print("bias true/prediction:{}/{}".format(bias, model.bias))
    print(model.predict(test))
    model.plot(x, y, 100)
