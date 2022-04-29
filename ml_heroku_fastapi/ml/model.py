from typing import List

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = svm.SVC(random_state=99)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slice_performance(df_test: pd.DataFrame, y_test: np.array, y_pred: np.array, categorical_features: List[str]) -> pd.DataFrame:
    """Computes the performance for each categorical feature of the test data.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test dataframe.
    y_test : np.array
        Encoded y data.
    y_pred : np.array
        Encoded y predicted data.
    categorical_features : List[str]
        List of categorical features.

    Returns
    -------
    pd.DataFrame
        Dataframe with values precision, recall, fbeta of each categorical slice.
    """
    df = df_test.copy()
    df["y_pred"] = y_pred
    df["y_test"] = y_test
    df_performance = pd.DataFrame(columns=["feature", "variable", "precision", "recall", "fbeta"])
    for cat_feat in categorical_features:
        for feat in df[cat_feat].unique():
            df_subsample = df[df[cat_feat] == feat]
            y_test = df_subsample["y_test"].to_numpy()
            y_pred = df_subsample["y_pred"].to_numpy()
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            df_performance = pd.concat([df_performance,
                                        pd.DataFrame.from_records([{"feature": cat_feat,
                                                                    "variable": feat,
                                                                    "precision": precision,
                                                                    "recall": recall,
                                                                    "fbeta": fbeta}])])
    return df_performance
