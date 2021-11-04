"""
TensorFlow Regression Example
"""
from numpy import linspace
from numpy.random import randn, randint
from pandas import concat, DataFrame
from matplotlib.pyplot import show, plot
from sklearn.model_selection import train_test_split
from tensorflow import (
    Variable,
    placeholder,
    float32,
    reduce_sum,
    square,
    compat,
    global_variables_initializer,
    Session,
    feature_column,
    estimator,
)


def wrapper():
    """
    wrapper function
    """
    x_data = linspace(0.0, 10.0, 1000000)
    noise = randn(len(x_data))
    y_true = (0.5 * x_data) + 5 + noise
    my_data = concat(
        [
            DataFrame(data=x_data, columns=["X Data"]),
            DataFrame(data=y_true, columns=["Y"]),
        ],
        axis=1,
    )
    print(my_data.head())
    my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")

    session(x_data, y_true, my_data)


def session(x_data, y_true, my_data):
    """
    Tensorflow session function
    """
    xph = placeholder(float32, [8])
    yph = placeholder(float32, [8])
    error = reduce_sum(square(yph - Variable(0.5) * xph + Variable(1.0)))
    optimizer = compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)
    mbs = [Variable(0.5), Variable(1.0)]

    with Session() as sess:
        sess.run(global_variables_initializer())
        batches = 1000
        for i in range(batches):
            print(i, batches, end="\r")
            rand_ind = randint(len(x_data), size=8)
            sess.run(train, feed_dict={xph: x_data[rand_ind], yph: y_true[rand_ind]})
        model_m, model_b = sess.run(mbs)

    models(model_m, model_b, x_data, y_true, my_data)


def models(model_m, model_b, x_data, y_true, my_data):
    """
    model function
    """
    print(model_m)
    print(model_b)
    y_hat = x_data * model_m + model_b
    my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
    plot(x_data, y_hat, "r")
    feat_cols = [feature_column.numeric_column("x", shape=[1])]
    estimators = estimator.LinearRegressor(feature_columns=feat_cols)

    x_train, x_eval, y_train, y_eval = train_test_split(
        x_data, y_true, test_size=0.3, random_state=101
    )
    print(x_train.shape)
    print(y_train.shape)
    print(x_eval.shape)
    print(y_eval.shape)

    input_func = estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True
    )
    train_input_func = estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False
    )
    eval_input_func = estimator.inputs.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False
    )

    estimate(estimators, input_func, train_input_func, eval_input_func, my_data)


def estimate(estimators, input_func, train_input_func, eval_input_func, my_data):
    """
    Set up Estimator Inputs Function
    """
    estimators.train(input_fn=input_func, steps=1000)
    train_metrics = estimators.evaluate(input_fn=train_input_func, steps=1000)
    eval_metrics = estimators.evaluate(input_fn=eval_input_func, steps=1000)
    print("train metrics: {}".format(train_metrics))
    print("eval metrics: {}".format(eval_metrics))

    input_fn_predict = estimator.inputs.numpy_input_fn(
        {"x": linspace(0, 10, 10)}, shuffle=False
    )
    print(list(estimators.predict(input_fn=input_fn_predict)))
    predictions = []
    for xda in estimators.predict(input_fn=input_fn_predict):
        predictions.append(xda["predictions"])
    print(predictions)

    my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
    plot(linspace(0, 10, 10), predictions, "r")


if __name__ == "__main__":
    wrapper()
    show()
