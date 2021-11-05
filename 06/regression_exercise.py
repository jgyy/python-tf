"""
TensorFlow Regression Exercise
"""
from os import environ
from os.path import join, dirname
from types import SimpleNamespace
from pandas import DataFrame, read_csv
from matplotlib.pyplot import show
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import compat, feature_column


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
    self.housing = DataFrame(read_csv(join(dirname(__file__), "cal_housing_clean.csv")))
    print(self.housing.head())
    print(self.housing.describe().transpose())
    self.x_data = self.housing.drop(["medianHouseValue"], axis=1)
    self.y_val = self.housing["medianHouseValue"]
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.x_data, self.y_val, test_size=0.3, random_state=101
    )
    self.scaler = MinMaxScaler()
    self.scaler.fit(self.X_train)
    self.X_train = DataFrame(
        data=self.scaler.transform(self.X_train),
        columns=self.X_train.columns,
        index=self.X_train.index,
    )
    self.X_test = DataFrame(
        data=self.scaler.transform(self.X_test),
        columns=self.X_test.columns,
        index=self.X_test.index,
    )
    print(self.housing.columns)
    age = feature_column.numeric_column("housingMedianAge")
    rooms = feature_column.numeric_column("totalRooms")
    bedrooms = feature_column.numeric_column("totalBedrooms")
    pop = feature_column.numeric_column("population")
    households = feature_column.numeric_column("households")
    income = feature_column.numeric_column("medianIncome")
    feat_cols = [age, rooms, bedrooms, pop, households, income]
    input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=self.X_train, y=self.y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    model = compat.v1.estimator.DNNRegressor(
        hidden_units=[6, 6, 6], feature_columns=feat_cols
    )
    model.train(input_fn=input_func, steps=25000)
    predict_input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=self.X_test, batch_size=10, num_epochs=1, shuffle=False
    )
    pred_gen = model.predict(predict_input_func)
    predictions = list(pred_gen)
    final_preds = []
    for pred in predictions:
        final_preds.append(pred["predictions"])
    print(mean_squared_error(self.y_test, final_preds) ** 0.5)

    return self


if __name__ == "__main__":
    main()
