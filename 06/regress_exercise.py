"""
Regression Exercise
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from tensorflow import feature_column, estimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def wrapper():
    """
    wrapper function
    """
    housing = DataFrame(read_csv(join(dirname(__file__), "cal_housing_clean.csv")))
    print(housing.head())
    print(housing.describe().transpose())
    x_data = housing.drop(["medianHouseValue"], axis=1)
    y_val = housing["medianHouseValue"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_val, test_size=0.3, random_state=101
    )
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = DataFrame(
        data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test = DataFrame(
        data=scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )
    print(housing.columns)
    age = feature_column.numeric_column("housingMedianAge")
    rooms = feature_column.numeric_column("totalRooms")
    bedrooms = feature_column.numeric_column("totalBedrooms")
    pop = feature_column.numeric_column("population")
    households = feature_column.numeric_column("households")
    income = feature_column.numeric_column("medianIncome")
    feat_cols = [age, rooms, bedrooms, pop, households, income]

    feature_columns(x_train, x_test, y_train, y_test, feat_cols)


def feature_columns(x_train, x_test, y_train, y_test, feat_cols):
    """
    feature columns function
    """
    input_func = estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    model = estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)
    model.train(input_fn=input_func, steps=25000)
    predict_input_func = estimator.inputs.pandas_input_fn(
        x=x_test, batch_size=10, num_epochs=1, shuffle=False
    )
    pred_gen = model.predict(predict_input_func)
    predictions = list(pred_gen)
    final_preds = []
    for pred in predictions:
        final_preds.append(pred["predictions"])
    print(mean_squared_error(y_test, final_preds) ** 0.5)


if __name__ == "__main__":
    wrapper()
