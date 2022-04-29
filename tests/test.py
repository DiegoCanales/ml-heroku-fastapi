import pytest
from ml_heroku_fastapi.ml.data import CensusDataset, process_data
from ml_heroku_fastapi.ml.model import train_model, compute_model_metrics, inference, slice_performance
import pandas as pd


@pytest.fixture(scope='session')
def data_raw():
    try:
        df = pd.read_csv(CensusDataset.sample_pth, sep=", ")
    except FileNotFoundError as err:
        raise err
    return df


@pytest.fixture(scope='session')
def data_preprocessed(data_raw):
    census = CensusDataset(data_raw)
    df_processed = census.preprocess()
    assert len(df_processed) > 0
    return df_processed


@pytest.fixture(scope='session')
def training(data_preprocessed):
    X_train, y_train, encoder, lb = process_data(X=data_preprocessed,
                                                 categorical_features=CensusDataset.CATEGORICAL_FEATURES,
                                                 label=CensusDataset.LABEL_FEATURE,
                                                 training=True)
    trained_model = train_model(X_train, y_train)
    return encoder, lb, trained_model


@pytest.fixture
def encode_data(training, data_preprocessed):
    encoder, lb, _ = training
    X, y, encoder, lb = process_data(X=data_preprocessed,
                                     categorical_features=CensusDataset.CATEGORICAL_FEATURES,
                                     label=CensusDataset.LABEL_FEATURE,
                                     training=False,
                                     encoder=encoder,
                                     lb=lb)
    return X, y, encoder, lb


def test_inference(encode_data, training):
    _, _, trained_model = training
    X, y, _, _ = encode_data
    y_pred = inference(trained_model, X)
    assert len(y_pred) == len(y)


def test_compute_model_metrics(encode_data, training):
    _, _, trained_model = training
    X, y, _, _ = encode_data

    y_pred = inference(trained_model, X)
    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    assert precision >= 0.0
    assert recall >= 0.0
    assert fbeta >= 0.0


def test_slice_performace(encode_data, training, data_preprocessed):
    _, _, trained_model = training
    X, y, _, _ = encode_data

    y_pred = inference(trained_model, X)
    df_slice_performance = slice_performance(data_preprocessed, y, y_pred,
                                             categorical_features=CensusDataset.CATEGORICAL_FEATURES)
    assert len(df_slice_performance) > 0
