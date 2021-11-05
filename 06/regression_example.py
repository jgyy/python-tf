"""
TensorFlow Regression Example
"""
from types import SimpleNamespace
from numpy import linspace
from numpy.random import randn, randint
from pandas import DataFrame, concat
from matplotlib.pyplot import plot, show
from sklearn.model_selection import train_test_split
from tensorflow import (
    float32,
    Variable,
    compat,
    feature_column,
    square,
    reduce_sum,
)


def decorator(function):
    """
    decorator function
    """

    def wrapper():
        compat.v1.disable_v2_behavior()
        self = function()
        if self:
            for items in self.__dict__.items():
                print(items)
        show()

    return wrapper


@decorator
def main(self=SimpleNamespace()):
    """
    main function
    """
    self.x_data = linspace(0.0, 10.0, 1000000)
    self.noise = randn(len(self.x_data))
    self.b = 5
    self.y_true = (0.5 * self.x_data) + 5 + self.noise
    self.my_data = concat(
        [
            DataFrame(data=self.x_data, columns=["X Data"]),
            DataFrame(data=self.y_true, columns=["Y"]),
        ],
        axis=1,
    )
    print(self.my_data.head())
    self.my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
    self.batch_size = 8
    self.m = Variable(0.5)
    self.b = Variable(1.0)
    self.xph = compat.v1.placeholder(float32, [self.batch_size])
    self.yph = compat.v1.placeholder(float32, [self.batch_size])
    self.y_model = self.m * self.xph + self.b
    self.error = reduce_sum(square(self.yph - self.y_model))
    self.optimizer = compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    self.train = self.optimizer.minimize(self.error)
    self.init = compat.v1.global_variables_initializer()
    with compat.v1.Session() as sess:
        sess.run(self.init)
        self.batches = 1000
        for _ in range(self.batches):
            self.rand_ind = randint(len(self.x_data), size=self.batch_size)
            self.feed = {
                self.xph: self.x_data[self.rand_ind],
                self.yph: self.y_true[self.rand_ind],
            }
            sess.run(self.train, feed_dict=self.feed)
        self.model_m, self.model_b = sess.run([self.m, self.b])
    print(self.model_m, self.model_b)
    self.y_hat = self.x_data * self.model_m + self.model_b
    self.my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
    plot(self.x_data, self.y_hat, "r")
    self.feat_cols = [feature_column.numeric_column("x", shape=[1])]
    self.estimator = compat.v1.estimator.LinearRegressor(self.feat_cols)
    self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(
        self.x_data, self.y_true, test_size=0.3, random_state=101
    )
    print(self.x_train.shape, self.y_train.shape, self.x_eval.shape, self.y_eval.shape)
    input_func = compat.v1.estimator.inputs.numpy_input_fn(
        {"x": self.x_train}, self.y_train, batch_size=4, num_epochs=None, shuffle=True
    )
    train_input_func = compat.v1.estimator.inputs.numpy_input_fn(
        {"x": self.x_train}, self.y_train, batch_size=4, num_epochs=1000, shuffle=False
    )
    eval_input_func = compat.v1.estimator.inputs.numpy_input_fn(
        {"x": self.x_eval}, self.y_eval, batch_size=4, num_epochs=1000, shuffle=False
    )
    self.estimator.train(input_fn=input_func, steps=1000)
    train_metrics = self.estimator.evaluate(input_fn=train_input_func, steps=1000)
    eval_metrics = self.estimator.evaluate(input_fn=eval_input_func, steps=1000)
    print("train metrics: {}".format(train_metrics))
    print("eval metrics: {}".format(eval_metrics))
    input_fn_predict = compat.v1.estimator.inputs.numpy_input_fn(
        {"x": linspace(0, 10, 10)}, shuffle=False
    )
    print(list(self.estimator.predict(input_fn=input_fn_predict)))
    predictions = []
    for x_data in self.estimator.predict(input_fn=input_fn_predict):
        predictions.append(x_data["predictions"])
    print(predictions)
    self.my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
    plot(linspace(0, 10, 10), predictions, "r")

    return self


if __name__ == "__main__":
    main()
