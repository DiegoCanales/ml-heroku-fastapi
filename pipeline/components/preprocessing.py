import pandas as pd
from ml_heroku_fastapi.ml.data import CensusDataset
from ml_heroku_fastapi.utils.config import logger


def run():
    logger.info(f"Reading data {CensusDataset.raw_pth}")
    df = pd.read_csv(CensusDataset.raw_pth, sep=", ")
    census = CensusDataset(df)
    logger.info("Preprocessing dataset")
    df = census.preprocess()
    logger.info("Dataset preprocessed")
    df.to_csv(CensusDataset.preprocessed_pth, sep=',', index=False)
    logger.info(f"Dataset saved at {CensusDataset.preprocessed_pth}")
