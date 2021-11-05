"""
TensorFlow Classification Example
"""
from os import environ
from os.path import join, dirname
from types import SimpleNamespace
from pandas import DataFrame, read_csv
from matplotlib.pyplot import show
from sklearn.model_selection import train_test_split
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
    self.diabetes = DataFrame(
        read_csv(join(dirname(__file__), "pima-indians-diabetes.csv"))
    )
    print(self.diabetes.head())
    print(self.diabetes.columns)
    self.cols_to_norm = [
        "Number_pregnant",
        "Glucose_concentration",
        "Blood_pressure",
        "Triceps",
        "Insulin",
        "BMI",
        "Pedigree",
    ]
    self.diabetes[self.cols_to_norm] = self.diabetes[self.cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    print(self.diabetes.head())
    print(self.diabetes.columns)
    self.num_preg = feature_column.numeric_column("Number_pregnant")
    self.plasma_gluc = feature_column.numeric_column("Glucose_concentration")
    self.dias_press = feature_column.numeric_column("Blood_pressure")
    self.tricep = feature_column.numeric_column("Triceps")
    self.insulin = feature_column.numeric_column("Insulin")
    self.bmi = feature_column.numeric_column("BMI")
    self.diabetes_pedigree = feature_column.numeric_column("Pedigree")
    self.age = feature_column.numeric_column("Age")
    self.assigned_group = feature_column.categorical_column_with_vocabulary_list(
        "Group", ["A", "B", "C", "D"]
    )
    self.diabetes["Age"].hist(bins=20)
    self.age_buckets = feature_column.bucketized_column(
        self.age, boundaries=[20, 30, 40, 50, 60, 70, 80]
    )
    self.feat_cols = [
        self.num_preg,
        self.plasma_gluc,
        self.dias_press,
        self.tricep,
        self.insulin,
        self.bmi,
        self.diabetes_pedigree,
        self.assigned_group,
        self.age_buckets,
    ]
    print(self.diabetes.head())
    print(self.diabetes.info())
    self.x_data = self.diabetes.drop("Class", axis=1)
    labels = self.diabetes["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        self.x_data, labels, test_size=0.33, random_state=101
    )
    input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    model = compat.v1.estimator.LinearClassifier(
        feature_columns=self.feat_cols, n_classes=2
    )
    model.train(input_fn=input_func, steps=1000)
    eval_input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False
    )
    results = model.evaluate(eval_input_func)
    print(results)
    pred_input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test, batch_size=10, num_epochs=1, shuffle=False
    )
    predictions = model.predict(pred_input_func)
    print(list(predictions))
    dnn_model = compat.v1.estimator.DNNClassifier(
        hidden_units=[10, 10, 10], feature_columns=self.feat_cols, n_classes=2
    )
    embedded_group_column = feature_column.embedding_column(
        self.assigned_group, dimension=4
    )
    feat_cols = [
        self.num_preg,
        self.plasma_gluc,
        self.dias_press,
        self.tricep,
        self.insulin,
        self.bmi,
        self.diabetes_pedigree,
        embedded_group_column,
        self.age_buckets,
    ]
    input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    dnn_model = compat.v1.estimator.DNNClassifier(
        hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2
    )
    dnn_model.train(input_fn=input_func, steps=1000)
    eval_input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False
    )
    print(dnn_model.evaluate(eval_input_func))

    return self


if __name__ == "__main__":
    main()
