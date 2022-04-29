import argparse

from ml_heroku_fastapi.utils.config import logger

from components import model_evaluation, preprocessing, train


PREPROCESSING = "preprocessing"
TRAIN = "train"
MODEL_EVALUATION = "model_evaluation"


_steps = [
    PREPROCESSING,
    TRAIN,
    MODEL_EVALUATION
]


def execute(args):
    steps = args.steps
    active_steps = steps.split(",") if steps != "all" else _steps

    if PREPROCESSING in active_steps:
        logger.info("RUNNING PREPROCESSING STEP")
        preprocessing.run()
    if TRAIN in active_steps:
        logger.info("RUNNING TRAIN STEP")
        train.run()
    if MODEL_EVALUATION in active_steps:
        logger.info("RUNNING MODEL_EVALUATION STEP")
        model_evaluation.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Component description",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--steps",
        type=str,
        help="Steps to execute separated by commas",
        default="all"
    )

    args = parser.parse_args()

    execute(args)
