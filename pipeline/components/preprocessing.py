import pandas as pd
from ml_heroku_fastapi.ml.data import CensusDataset
from ml_heroku_fastapi.utils.paths import DATA_INTERIM_DIR, DATA_RAW_DIR

CENSUS_DATA_PTH = DATA_RAW_DIR / "census.csv"
CENSUS_INTERIM = DATA_INTERIM_DIR / "census_preprocessed.csv"


def run():
    df = pd.read_csv(CENSUS_DATA_PTH, sep=", ")
    census = CensusDataset(df)
    df = census.preprocess()
    DATA_INTERIM_DIR.mkdir(exist_ok=True)
    df.to_csv(CENSUS_INTERIM, sep=',', index=False)
