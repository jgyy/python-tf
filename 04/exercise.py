"""
Crash Course Review Exercise
"""
from numpy.random import seed, randint
from matplotlib.pyplot import figure, imshow, colorbar, title, show
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def wrapper():
    """
    wrapper function
    """
    seed(101)
    mat = randint(1, 101, (100, 5))
    print(mat)
    figure()
    imshow(mat, aspect=0.05)
    colorbar()
    title("My Plot")
    daf = DataFrame(mat)
    print(daf)
    daf.plot(x=0, y=1, kind="scatter")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daf)
    print(scaled_data)
    daf.columns = ["f1", "f2", "f3", "f4", "label"]
    xda = daf[["f1", "f2", "f3", "f4"]]
    yda = daf["label"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda)
    print("x_train\n", x_train)
    print("x_test\n", x_test)
    print("y_train\n", y_train)
    print("y_test\n", y_test)


if __name__ == "__main__":
    wrapper()
    show()
