"""
TensorFlow Regression Exercise
"""
from os import environ
from os.path import join, dirname
from types import SimpleNamespace
from pandas import DataFrame, read_csv
from matplotlib.pyplot import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
    self.census = DataFrame(read_csv(join(dirname(__file__), "census_data.csv")))
    print(self.census.head())
    self.census["income_bracket"].unique()

    def label_fix(label):
        if label == " <=50K":
            return 0
        return 1

    self.census["income_bracket"] = self.census["income_bracket"].apply(label_fix)
    self.x_data = self.census.drop("income_bracket", axis=1)
    self.y_labels = self.census["income_bracket"]
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.x_data, self.y_labels, test_size=0.3, random_state=101
    )
    print(self.census.columns)
    self.gender = feature_column.categorical_column_with_vocabulary_list(
        "gender", ["Female", "Male"]
    )
    self.occupation = feature_column.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000
    )
    self.marital_status = feature_column.categorical_column_with_hash_bucket(
        "marital_status", hash_bucket_size=1000
    )
    self.relationship = feature_column.categorical_column_with_hash_bucket(
        "relationship", hash_bucket_size=1000
    )
    self.education = feature_column.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=1000
    )
    self.workclass = feature_column.categorical_column_with_hash_bucket(
        "workclass", hash_bucket_size=1000
    )
    self.native_country = feature_column.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000
    )
    self.age = feature_column.numeric_column("age")
    self.education_num = feature_column.numeric_column("education_num")
    self.capital_gain = feature_column.numeric_column("capital_gain")
    self.capital_loss = feature_column.numeric_column("capital_loss")
    self.hours_per_week = feature_column.numeric_column("hours_per_week")
    feat_cols = [
        self.gender,
        self.occupation,
        self.marital_status,
        self.relationship,
        self.education,
        self.workclass,
        self.native_country,
        self.age,
        self.education_num,
        self.capital_gain,
        self.capital_loss,
        self.hours_per_week,
    ]
    input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=self.X_train, y=self.y_train, batch_size=100, num_epochs=None, shuffle=True
    )
    model = compat.v1.estimator.LinearClassifier(feature_columns=feat_cols)
    model.train(input_fn=input_func, steps=5000)
    pred_fn = compat.v1.estimator.inputs.pandas_input_fn(
        x=self.X_test, batch_size=len(self.X_test), shuffle=False
    )
    predictions = list(model.predict(input_fn=pred_fn))
    print(predictions[0])
    final_preds = []
    for pred in predictions:
        final_preds.append(pred["class_ids"][0])
    print(final_preds[:10])
    print(classification_report(self.y_test, final_preds))

    return self


if __name__ == "__main__":
    main()
