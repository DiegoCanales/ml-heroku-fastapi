import os

import pandas as pd
from fastapi import FastAPI
from joblib import load
from ml_heroku_fastapi.ml.data import CensusDataset, process_data
from ml_heroku_fastapi.ml.model import inference
from ml_heroku_fastapi.utils.paths import MODEL_DIR
from pydantic import BaseModel, Field


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = load(MODEL_DIR / "model.joblib")
encoder = load(MODEL_DIR / "encoder.joblib")
lb = load(MODEL_DIR / "lb.joblib")


class CensusModel(BaseModel):
    workclass: str = Field(alias='workclass')
    education: str = Field(alias='education')
    marital_status: str = Field(alias='marital-status')
    occupation: str = Field(alias='occupation')
    race: str = Field(alias='race')
    sex: str = Field(alias='sex')
    native_country: str = Field(alias='native-country')
    age: int = Field(alias='age')
    hours_per_week: int = Field(alias='hours-per-week')

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                'workclass': 'State-gov',
                'education': 'HS-grad',
                'marital-status': 'Divorced',
                'occupation': 'Adm-clerical',
                'race': 'White',
                'sex': 'Female',
                'native-country': 'United-States',
                'age': 50,
                'hours-per-week': 46
            }
        }


# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def greeting():
    return {"greeting": "Welcome!"}


@app.post("/inference")
async def model_inference(sample: CensusModel):
    df_sample = pd.DataFrame(sample.dict(by_alias=True), index=[0])
    X, _, _, _ = process_data(X=df_sample,
                              categorical_features=CensusDataset.CATEGORICAL_FEATURES,
                              training=False,
                              encoder=encoder,
                              lb=lb)
    y_pred = inference(model, X)
    preds = lb.inverse_transform(y_pred)
    return {"pred": list(preds)}
