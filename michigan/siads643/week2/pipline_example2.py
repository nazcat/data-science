import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer


def make_pipeline():
    # Replace this and the line below with your code.
    # The function should return a sklearn pipeline.

    numeric_features = ["permit_lot_size"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='median')),
            ("scalar", StandardScaler())
            ]
        )
    categorical_features = ["permit_type","permit_subtype"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ]
        )
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                            ("regressor", RandomForestRegressor())])

    return pipe
    # raise NotImplementedError


def train(data_frame):
    # We are predicting the wait time
    y = (data_frame["issued_date"] - data_frame["submitted_date"]).dt.days

    # Drop columns in dataframe that shouldn't be used to predict wait time
    x = data_frame.drop(
        [
            "issued_date",
            "submitted_date",
        ],
        axis=1,
    )

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    trained_model = make_pipeline()

    trained_model.fit(x_train, y_train)

    # Store model metrics in a dictionary
    model_metrics = {
        "train_data": {
            "score": trained_model.score(x_train, y_train),
            "mae": mean_absolute_error(y_train, trained_model.predict(x_train)),
        },
        "test_data": {
            "score": trained_model.score(x_test, y_test),
            "mae": mean_absolute_error(y_test, trained_model.predict(x_test)),
        },
    }
    print(model_metrics)

    return trained_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="cleaned data file (CSV)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display metrics",
    )
    args = parser.parse_args()

    input_data = pd.read_csv(
        args.input_file,
        parse_dates=["submitted_date", "issued_date"],
    )

    model = train(input_data)
