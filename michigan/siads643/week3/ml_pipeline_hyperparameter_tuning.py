###
Machine Learning - Pipeline with hypeparameter rtuning 
###

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


def make_pipeline():
    # Replace this and the line below with your pipeline
    # from the previous assignment

    numeric_features = ["permit_lot_size"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='mean')),
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


def tune_hyperparameters(pipe, x_train, y_train):
    """
    Tune the hyperparameters of your pipeline using GridSearchCV.
    :param pipe: sklearn pipeline
    :param x_train: training data
    :param y: training target
    :return: GridSearchCV object that has been fit to your training data
    """
    # Replace this function with your hyperparameter tuning code.
    grid_params = {
        "preprocessor__num__imputer__strategy": ['mean', 'median'],
        "regressor__n_estimators": [10, 100, 1000]}
    clf = GridSearchCV(pipe, grid_params, cv = 5)

    return clf.fit(x_train, y_train)

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

    # Make the pipeline
    pipe = make_pipeline()

    # Hyperparameter tune the pipeline on your training set and get the best model
    best_model = tune_hyperparameters(pipe, x_train, y_train)

    # Get score on the test set
    y_pred = best_model.predict(x_test)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    print(best_model_mae)
    return best_model


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

    best_result = train(input_data)
