"""
TensorFlow Classification
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from tensorflow import feature_column, estimator
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split


def wrapper():
    """
    wrapper function
    """
    diabetes = DataFrame(read_csv(join(dirname(__file__), "pima-indians-diabetes.csv")))
    print(diabetes.head())
    print(diabetes.columns)
    cols_to_norm = [
        "Number_pregnant",
        "Glucose_concentration",
        "Blood_pressure",
        "Triceps",
        "Insulin",
        "BMI",
        "Pedigree",
    ]
    diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    print(diabetes.head())
    print(diabetes.columns)
    num_preg = feature_column.numeric_column("Number_pregnant")
    plasma_gluc = feature_column.numeric_column("Glucose_concentration")
    dias_press = feature_column.numeric_column("Blood_pressure")
    tricep = feature_column.numeric_column("Triceps")
    insulin = feature_column.numeric_column("Insulin")
    bmi = feature_column.numeric_column("BMI")
    diabetes_pedigree = feature_column.numeric_column("Pedigree")
    age = feature_column.numeric_column("Age")
    assigned_group = feature_column.categorical_column_with_vocabulary_list(
        "Group", ["A", "B", "C", "D"]
    )

    figure()
    diabetes["Age"].hist(bins=20)
    age_buckets = feature_column.bucketized_column(
        age, boundaries=[20, 30, 40, 50, 60, 70, 80]
    )
    feat_cols = [
        num_preg,
        plasma_gluc,
        dias_press,
        tricep,
        insulin,
        bmi,
        diabetes_pedigree,
        assigned_group,
        age_buckets,
    ]
    print(diabetes.head())
    print(diabetes.info())

    tts(feat_cols, diabetes, assigned_group)


def tts(feat_cols, diabetes, assigned_group):
    """
    train test split function
    """
    x_train, x_test, y_train, y_test = train_test_split(
        diabetes.drop("Class", axis=1),
        diabetes["Class"],
        test_size=0.33,
        random_state=101,
    )
    input_func = estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    model = estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
    model.train(input_fn=input_func, steps=1000)
    eval_input_func = estimator.inputs.pandas_input_fn(
        x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False
    )
    results = model.evaluate(eval_input_func)
    print(results)
    pred_input_func = estimator.inputs.pandas_input_fn(
        x=x_test, batch_size=10, num_epochs=1, shuffle=False
    )
    predictions = model.predict(pred_input_func)
    print(list(predictions))
    embedded_group_column = feature_column.embedding_column(assigned_group, dimension=4)
    feat_cols[7] = embedded_group_column
    input_func = estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
    )
    dnn_model = estimator.DNNClassifier(
        hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2
    )
    dnn_model.train(input_fn=input_func, steps=1000)
    eval_input_func = estimator.inputs.pandas_input_fn(
        x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False
    )
    print(dnn_model.evaluate(eval_input_func))


if __name__ == "__main__":
    wrapper()
    show()
