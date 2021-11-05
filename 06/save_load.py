"""
Saving and Loading Models
"""
from os import environ
from os.path import join, dirname
from types import SimpleNamespace
from numpy import linspace
from numpy.random import uniform, rand
from matplotlib.pyplot import show, plot, figure
from tensorflow import Variable, compat, reduce_mean


def decorator(function):
    """
    decorator function
    """

    def wrapper():
        environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        compat.v1.disable_v2_behavior()
        self = function()
        for items in self.__dict__.items():
            print(items)
        show()

    return wrapper


@decorator
def main(self=SimpleNamespace()):
    """
    main function
    """
    self.x_data = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    print(self.x_data)
    self.y_label = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    figure()
    plot(self.x_data, self.y_label, "*")
    print(rand(2))
    self.m = Variable(0.39)
    self.b = Variable(0.2)
    error = reduce_mean(self.y_label - (self.m * self.x_data + self.b))
    optimizer = compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)
    init = compat.v1.global_variables_initializer()
    saver = compat.v1.train.Saver()
    with compat.v1.Session() as sess:
        sess.run(init)
        epochs = 100
        for _ in range(epochs):
            sess.run(train)
        final_slope, final_intercept = sess.run([self.m, self.b])
        saver.save(sess, join(dirname(__file__), "new_models/my_second_model.ckpt"))
    x_test = linspace(-1, 11, 10)
    y_pred_plot = final_slope * x_test + final_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(self.x_data, self.y_label, "*")
    with compat.v1.Session() as sess:
        saver.restore(sess, join(dirname(__file__), "new_models/my_second_model.ckpt"))
        restored_slope, restored_intercept = sess.run([self.m, self.b])
    x_test = linspace(-1, 11, 10)
    y_pred_plot = restored_slope * x_test + restored_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(self.x_data, self.y_label, "*")

    return self


if __name__ == "__main__":
    main()
