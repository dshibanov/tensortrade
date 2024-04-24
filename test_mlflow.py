import os
import tempfile
import time

import mlflow

from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow


def evaluation_fn(step, width, height):

    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def train_function(config):
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back to Tune.
        train.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(10)



def tune_with_callback(mlflow_tracking_uri, finish_fast=False):
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(num_samples=5),
        run_config=train.RunConfig(
            name="my_mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name="mlflow_callback_example666",
                    save_artifact=True,
                )
            ],
        ),
        param_space={
            "width": tune.randint(10, 100),
            "height": tune.randint(0, 100),
            "steps": 5 if finish_fast else 100,
        },
    )
    results = tuner.fit()


# def train_function_mlflow(config):
#     tracking_uri = config.pop("tracking_uri", None)
#     setup_mlflow(
#         config,
#         experiment_name="setup_mlflow_example",
#         tracking_uri=tracking_uri,
#     )

#     # Hyperparameters
#     width, height = config["width"], config["height"]

#     for step in range(config.get("steps", 100)):
#         # Iterative training function - can be any arbitrary training procedure
#         intermediate_score = evaluation_fn(step, width, height)
#         # Log the metrics to mlflow
#         mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
#         # Feed the score back to Tune.
#         train.report({"iterations": step, "mean_loss": intermediate_score})
#         time.sleep(0.1)


# def tune_with_setup(mlflow_tracking_uri, finish_fast=False):
#     # Set the experiment, or create a new one if does not exist yet.
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     mlflow.set_experiment(experiment_name="setup_mlflow_example")

#     tuner = tune.Tuner(
#         train_function_mlflow,
#         tune_config=tune.TuneConfig(num_samples=5),
#         run_config=train.RunConfig(
#             name="mlflow",
#         ),
#         param_space={
#             "width": tune.randint(10, 100),
#             "height": tune.randint(0, 100),
#             "steps": 5 if finish_fast else 100,
#             "tracking_uri": mlflow.get_tracking_uri(),
#         },
#     )
#     results = tuner.fit()

def test1():

    # smoke_test = True
    smoke_test = False

    if smoke_test:
        mlflow_tracking_uri = os.path.join(tempfile.gettempdir(), "mlruns")
    else:
        mlflow_tracking_uri = "<MLFLOW_TRACKING_URI>"

    import icecream as ic
    uc.enable()

    ic(mlflow_tracking_uri)

    tune_with_callback(mlflow_tracking_uri, finish_fast=smoke_test)
    if not smoke_test:
        df = mlflow.search_runs(
            [mlflow.get_experiment_by_name("mlflow_callback_example666").experiment_id]
        )
        print(df)

    # tune_with_setup(mlflow_tracking_uri, finish_fast=smoke_test)
    # if not smoke_test:
    #     df = mlflow.search_runs(
    #         [mlflow.get_experiment_by_name("setup_mlflow_example").experiment_id]
    #     )
    #     print(df)



def test2():

    import mlflow
    from mlflow.models import infer_signature

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

if __name__ == "__main__":
    test1()
    # test2()
