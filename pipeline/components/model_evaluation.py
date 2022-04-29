import pandas as pd
from joblib import load
from ml_heroku_fastapi.ml.data import CensusDataset, process_data
from ml_heroku_fastapi.ml.model import compute_model_metrics, inference, slice_performance
from ml_heroku_fastapi.utils.paths import METRICS_DIR, MODEL_DIR
from ml_heroku_fastapi.utils.config import logger


def run():
    logger.info(f"Reading data {CensusDataset.test_pth}")
    df_test = pd.read_csv(CensusDataset.test_pth)
    model = load(MODEL_DIR / "model.joblib")
    encoder = load(MODEL_DIR / "encoder.joblib")
    lb = load(MODEL_DIR / "lb.joblib")

    logger.info(f"Reading model {MODEL_DIR / 'model.joblib'}")
    logger.info(f"Reading encoder {MODEL_DIR / 'encoder.joblib'}")
    logger.info(f"Reading label encoder {MODEL_DIR / 'lb.joblib'}")

    X_test, y_test, encoder, lb = process_data(X=df_test,
                                               categorical_features=CensusDataset.CATEGORICAL_FEATURES,
                                               label=CensusDataset.LABEL_FEATURE,
                                               training=False,
                                               encoder=encoder,
                                               lb=lb)

    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    df_metrics = pd.DataFrame.from_dict({"precision": [precision], "recall": [recall], "fbeta": [fbeta]})
    df_metrics.to_csv(METRICS_DIR / "general.csv", index=False)
    logger.info(f"Model metrics saved at {METRICS_DIR / 'general.csv'}")

    logger.info("Computing slice performance")
    df_slice_performance = slice_performance(df_test, y_test, y_pred,
                                             categorical_features=CensusDataset.CATEGORICAL_FEATURES)
    df_slice_performance.to_csv(METRICS_DIR / "slice_performance.csv", index=False)
    logger.info(f"Slice performance saved at {METRICS_DIR / 'slice_performance.csv'}")
