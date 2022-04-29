from fastapi.testclient import TestClient
from joblib import load
from ml_heroku_fastapi.utils.paths import MODEL_DIR

from main import app

lb = load(MODEL_DIR / "lb.joblib")
client = TestClient(app)


def test_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome!"}


def test_inference_class_zero():
    data = {'workclass': 'State-gov', 'education': 'HS-grad', 'marital-status': 'Divorced', 'occupation': 'Adm-clerical',
            'race': 'White', 'sex': 'Female', 'native-country': 'United-States', 'age': 50, 'hours-per-week': 46}
    response = client.post("/inference",
                           json=data)
    assert response.status_code == 200
    assert response.json() == {"pred": [lb.classes_[0]]}


def test_inference_class_one():
    data = {'workclass': 'Private', 'education': 'Bachelors', 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
            'race': 'White', 'sex': 'Male', 'native-country': 'United-States', 'age': 49, 'hours-per-week': 66}
    response = client.post("/inference",
                           json=data)
    assert response.status_code == 200
    assert response.json() == {"pred": [lb.classes_[1]]}
