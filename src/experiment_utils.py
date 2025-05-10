import logging
import os

import mlflow
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI and credentials set.")

    return mlflow


def log_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="metric",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow.

    Parameters:
    - model: Trained model object (e.g., sklearn model).
    - input_data: Input data used for training (for signature inference).
    - experiment_name: Name of the MLflow experiment.
    - metric_name: Name of the metric to log (e.g., "RMSE", "accuracy").
    - model_name: Optional name for the registered model.
    - params: Optional dictionary of hyperparameters to log.
    - score: Optional evaluation metric to log.
    """
    try:
        # Set the experiment
        uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(uri)
        logger.info("MLflow tracking URI and credentials set.")
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        # Start an MLflow run
        with mlflow.start_run():
            # Log hyperparameters if provided
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {params}")

            # Log metrics if provided
            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            # Infer the model signature
            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("Model signature inferred.")

            # Determine the model name
            if not model_name:
                model_name = model.__class__.__name__

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )
            logger.info(f"Model logged with name: {model_name}")
            return model_info

    except Exception as e:
        logger.error(f"An error occurred while logging to MLflow: {e}")
        raise
