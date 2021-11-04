"""
Classification Exercise
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from tensorflow import feature_column, estimator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def wrapper():
    """
    wrapper function
    """
    census = DataFrame(read_csv(join(dirname(__file__), "census_data.csv")))
    print(census.head())
    print(census["income_bracket"].unique())
    label_fix = lambda label: 0 if label == " <=50K" else 1
    census["income_bracket"] = census["income_bracket"].apply(label_fix)
    x_data = census.drop("income_bracket", axis=1)
    y_labels = census["income_bracket"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_labels, test_size=0.3, random_state=101
    )
    print(census.columns)

    feat_cols = feature_columns()
    inputs(x_train, x_test, y_train, y_test, feat_cols)


def feature_columns():
    """
    Create the Feature Columns for tensorflow estimator
    """
    gender = feature_column.categorical_column_with_vocabulary_list(
        "gender", ["Female", "Male"]
    )
    occupation = feature_column.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000
    )
    marital_status = feature_column.categorical_column_with_hash_bucket(
        "marital_status", hash_bucket_size=1000
    )
    relationship = feature_column.categorical_column_with_hash_bucket(
        "relationship", hash_bucket_size=1000
    )
    education = feature_column.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=1000
    )
    workclass = feature_column.categorical_column_with_hash_bucket(
        "workclass", hash_bucket_size=1000
    )
    native_country = feature_column.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000
    )
    age = feature_column.numeric_column("age")
    education_num = feature_column.numeric_column("education_num")
    capital_gain = feature_column.numeric_column("capital_gain")
    capital_loss = feature_column.numeric_column("capital_loss")
    hours_per_week = feature_column.numeric_column("hours_per_week")
    feat_cols = [
        gender,
        occupation,
        marital_status,
        relationship,
        education,
        workclass,
        native_country,
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]
    return feat_cols


def inputs(x_train, x_test, y_train, y_test, feat_cols):
    """
    Create Input Function
    """
    input_func = estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True
    )
    model = estimator.LinearClassifier(feature_columns=feat_cols)
    model.train(input_fn=input_func, steps=5000)
    pred_fn = estimator.inputs.pandas_input_fn(
        x=x_test, batch_size=len(x_test), shuffle=False
    )
    predictions = list(model.predict(input_fn=pred_fn))
    print(predictions[0])
    final_preds = []
    for pred in predictions:
        final_preds.append(pred["class_ids"][0])
    print(final_preds[:10])
    print(classification_report(y_test, final_preds))


if __name__ == "__main__":
    wrapper()
