"""
First Neurons
"""
from numpy import linspace
from numpy.random import seed, uniform, random, rand
from matplotlib.pyplot import figure, plot, show
from tensorflow import (
    set_random_seed,
    placeholder,
    float32,
    Session,
    Variable,
    zeros,
    random_normal,
    matmul,
    add,
    sigmoid,
    global_variables_initializer,
    compat,
)


def wrapper():
    """
    wrapper function
    """
    seed(101)
    set_random_seed(101)
    rand_a = uniform(0, 100, (5, 5))
    print(rand_a)
    rand_b = uniform(0, 100, (5, 1))
    print(rand_b)
    seed(101)
    rand_a = uniform(0, 100, (5, 5))
    rand_b = uniform(0, 100, (5, 1))
    ada = placeholder(float32)
    bda = placeholder(float32)
    add_op = ada + bda
    mult_op = ada * bda

    with Session() as sess:
        add_result = sess.run(add_op, feed_dict={ada: rand_a, bda: rand_b})
        print(add_result)
        print("\n")
        mult_result = sess.run(mult_op, feed_dict={ada: rand_a, bda: rand_b})
        print(mult_result)

    nn_func()


def nn_func():
    """
    Example Neural Network function
    """
    n_features = 10
    n_dense_neurons = 3
    xda = placeholder(float32, (None, n_features))
    bda = Variable(zeros([n_dense_neurons]))
    wda = Variable(random_normal([n_features, n_dense_neurons]))
    xwd = matmul(xda, wda)
    zda = add(xwd, bda)
    ada = sigmoid(zda)
    init = global_variables_initializer()

    with Session() as sess:
        sess.run(init)
        layer_out = sess.run(ada, feed_dict={xda: random([1, n_features])})

    print(layer_out)
    x_data = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    print(x_data)
    y_label = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    figure()
    plot(x_data, y_label, "*")
    print(rand(2))
    mda = Variable(0.39)
    bda = Variable(0.2)

    cost_func(x_data, y_label, mda, bda)


def cost_func(x_data, y_label, mda, bda):
    """
    Cost function
    """
    error = 0
    for xda, yda in zip(x_data, y_label):
        y_hat = mda * xda + bda
        error += (yda - y_hat) ** 2

    optimizer = compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    trains = optimizer.minimize(error)

    with Session() as sess:
        sess.run(global_variables_initializer())
        epochs = 100
        for _ in range(epochs):
            sess.run(trains)
        final_slope, final_intercept = sess.run([mda, bda])

    session(final_slope, final_intercept, x_data, y_label)


def session(final_slope, final_intercept, x_data, y_label):
    """
    Create Session and Run Function
    """
    print(final_slope)
    print(final_intercept)
    x_test = linspace(-1, 11, 10)
    y_pred_plot = final_slope * x_test + final_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(x_data, y_label, "*")


if __name__ == "__main__":
    wrapper()
    show()
