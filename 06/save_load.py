"""
Saving and Loading Models
"""
from os.path import join, dirname
from numpy import linspace
from numpy.random import seed, uniform, rand
from matplotlib.pyplot import plot, figure, show
from tensorflow import (
    set_random_seed,
    Variable,
    reduce_mean,
    train,
    global_variables_initializer,
    Session,
)

MODEL = join(dirname(__file__), "new_models/my_second_model.ckpt")


def wrapper():
    """
    wrapper function
    """
    seed(101)
    set_random_seed(101)
    x_data = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    print(x_data)
    y_label = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    figure()
    plot(x_data, y_label, "*")
    print(rand(2))
    mda = Variable(0.39)
    bda = Variable(0.2)
    error = reduce_mean(y_label - (mda * x_data + bda))
    optimizer = train.GradientDescentOptimizer(learning_rate=0.001)
    trains = optimizer.minimize(error)
    saver = train.Saver()

    with Session() as sess:
        sess.run(global_variables_initializer())
        epochs = 100
        for _ in range(epochs):
            sess.run(trains)
        final_slope, final_intercept = sess.run([mda, bda])
        saver.save(sess, MODEL)

    x_test = linspace(-1, 11, 10)
    y_pred_plot = final_slope * x_test + final_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(x_data, y_label, "*")

    load_model(saver, mda, bda, x_data, y_label)


def load_model(saver, mda, bda, x_data, y_label):
    """
    Loading a Model Function
    """
    with Session() as sess:
        saver.restore(sess, MODEL)
        restored_slope, restored_intercept = sess.run([mda, bda])
    x_test = linspace(-1, 11, 10)
    y_pred_plot = restored_slope * x_test + restored_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(x_data, y_label, "*")


if __name__ == "__main__":
    wrapper()
    show()
