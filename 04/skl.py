"""
SciKit Learn Preprocessing Overview
"""
from numpy.random import randint
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def wrapper():
    """
    wrapper function
    """
    data = randint(0, 100, (10, 2))
    print(data)
    scaler_model = MinMaxScaler()
    scaler_model.fit(data)
    result = scaler_model.fit_transform(data)
    print(result)
    data = DataFrame(data=randint(0, 101, (50, 4)), columns=["f1", "f2", "f3", "label"])
    print(data.head())
    xda = data[["f1", "f2", "f3"]]
    yda = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        xda, yda, test_size=0.3, random_state=101
    )
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


if __name__ == "__main__":
    wrapper()
