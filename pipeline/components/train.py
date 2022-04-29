import pandas as pd
from joblib import dump
from ml_heroku_fastapi.ml.data import CensusDataset, process_data
from ml_heroku_fastapi.ml.model import train_model
from ml_heroku_fastapi.utils.config import logger
from ml_heroku_fastapi.utils.paths import MODEL_DIR
from sklearn.model_selection import train_test_split


def run():
    df = pd.read_csv(CensusDataset.preprocessed_pth)
    logger.info(f"Data readed {CensusDataset.preprocessed_pth}")
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=7)
    logger.info("Data splitted")
    X_train, y_train, encoder, lb = process_data(X=df_train,
                                                 categorical_features=CensusDataset.CATEGORICAL_FEATURES,
                                                 label=CensusDataset.LABEL_FEATURE,
                                                 training=True)
    logger.info("Data encoded")
    logger.info("Start training")
    trained_model = train_model(X_train, y_train)
    logger.info("Training finished")
    logger.info("Saving model artifacts at:")
    dump(trained_model, MODEL_DIR / "model.joblib")
    dump(encoder, MODEL_DIR / "encoder.joblib")
    dump(lb, MODEL_DIR / "lb.joblib")
    df_test.to_csv(CensusDataset.test_pth, sep=',', index=False)
    df_train.to_csv(CensusDataset.train_pth, sep=',', index=False)

    logger.info(f"{MODEL_DIR / 'model.joblib'}")
    logger.info(f"{MODEL_DIR / 'encoder.joblib'}")
    logger.info(f"{MODEL_DIR / 'lb.joblib'}")
    logger.info(f"Test data saved at: {CensusDataset.test_pth}")
    logger.info(f"Train data saved at: {CensusDataset.train_pth}")
