"""
First Neurons
"""
from types import SimpleNamespace
from numpy import linspace
from numpy.random import uniform, random
from matplotlib.pyplot import plot, show, figure
from tensorflow import float32, Variable, compat, zeros, matmul, add, sigmoid


def main():
    """
    main function
    """
    sns = SimpleNamespace()
    sns.rand_a = uniform(0, 100, (5, 5))
    sns.rand_b = uniform(0, 100, (5, 1))
    sns.a = compat.v1.placeholder(float32)
    sns.b = compat.v1.placeholder(float32)
    sns.add_op = sns.a + sns.b
    sns.mult_op = sns.a * sns.b
    with compat.v1.Session() as sess:
        sns.add_result = sess.run(
            sns.add_op, feed_dict={sns.a: sns.rand_a, sns.b: sns.rand_b}
        )
        sns.mult_result = sess.run(
            sns.mult_op, feed_dict={sns.a: sns.rand_a, sns.b: sns.rand_b}
        )
    sns.n_features = 10
    sns.n_dense_neurons = 3
    sns.x = compat.v1.placeholder(float32, (None, sns.n_features))
    sns.b = Variable(zeros([sns.n_dense_neurons]))
    sns.W = Variable(compat.v1.random_normal([sns.n_features, sns.n_dense_neurons]))
    sns.xW = matmul(sns.x, sns.W)
    sns.z = add(sns.xW, sns.b)
    sns.a = sigmoid(sns.z)
    sns.init = compat.v1.global_variables_initializer
    with compat.v1.Session() as sess:
        sess.run(sns.init())
        sns.layer_out = sess.run(sns.a, feed_dict={sns.x: random([1, sns.n_features])})
    sns.x_data = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    sns.y_label = linspace(0, 10, 10) + uniform(-1.5, 1.5, 10)
    plot(sns.x_data, sns.y_label, "*")
    sns.m = Variable(0.39)
    sns.b = Variable(0.2)
    sns.error = 0
    for attr in sns.__dict__.items():
        print(attr)
    for x_var, y_var in zip(sns.x_data, sns.y_label):
        y_hat = sns.m * x_var + sns.b
        sns.error += (y_var - y_hat) ** 2
    optimizer = compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(sns.error)
    with compat.v1.Session() as sess:
        sess.run(sns.init())
        epochs = 100
        for _ in range(epochs):
            sess.run(train)
        final_slope, final_intercept = sess.run([sns.m, sns.b])
    print(final_slope)
    print(final_intercept)
    x_test = linspace(-1, 11, 10)
    y_pred_plot = final_slope * x_test + final_intercept
    figure()
    plot(x_test, y_pred_plot, "r")
    plot(sns.x_data, sns.y_label, "*")


if __name__ == "__main__":
    compat.v1.disable_v2_behavior()
    main()
    show()
