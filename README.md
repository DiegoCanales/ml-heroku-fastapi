# Udacity MLOps Nanodegree Project: Deploying a Machine Learning Model on Heroku with FastAPI

The goal of the project is develop a classification model on publicly available Census Bureau data. Create the pipeline, deploy an API developed with FastAPI and deploy to Heroku.

## Getting Started

To run the entire project you must create the conda environment.

```bash
conda env create -f environment.yaml
```

### Get the data

```bash
dvc pull
```

### EDA

The EDA notebook is located at `notebooks/eda.ipynb` and the profiling report is located at `reports/eda_profiling.html`.

### Pipeline

To execute the pipeline must run the following commands.

```bash
cd pipeline
python main.py
```

### DVC pipeline

To execute the dvc pipeline execute:

```bash
dvc repro
```

### Tests

To execute the tests must run the following command.

```bash
pytest -s tests/*
```

### API

Run the API.

```bash
cd api
uvicorn --reload --app-dir api/ main:app
```

#### Test API

```bash
pytest -s api/test_api.py
```

#### Request live API

To request the live API, execute the following command:

```bash
python api/request_api.py
```