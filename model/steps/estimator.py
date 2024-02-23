# https://sagemaker.readthedocs.io/en/stable/index.html
import os
from pathlib import Path

from sagemaker.huggingface import HuggingFace
from comet_ml.integration.sagemaker import log_sagemaker_training_job_v1

from config import EstimatorConfig, ModelTrainingConfig


class LayoutLMEstimator(HuggingFace):
    """Custom LayoutLM SagemakerEstimator"""

    def __init__(self):
        """Sagemaker estimator."""

        hyperparameters = {
            "epochs": ModelTrainingConfig.epochs,
            "per_device_train_batch_size": ModelTrainingConfig.per_device_train_batch_size,
            "per_device_eval_batch_size": ModelTrainingConfig.per_device_eval_batch_size,
            "lr": ModelTrainingConfig.lr,
            "pretrained_model_name": ModelTrainingConfig.pretrained_model_name,
            "labels": " ".join(f"'{label}'" for label in ModelTrainingConfig.labels),
        }

        # Metrics returned by the Trainer and tracked by SageMaker during training
        # Warning: the log frequency is about 1min, whihc means logs are lost
        metrics_defintions = [
            {'Name': 'loss', 'Regex': "'loss': (.*?),"},
            {'Name': 'eval_loss', 'Regex': "'eval_loss': (.*?),"},
            {'Name': 'learning_rate', 'Regex': "'learning_rate': (.*?),"},
            {'Name': 'eval_overall_recall', 'Regex': "'eval_overall_recall': (.*?),"},
            {'Name': 'eval_overall_f1', 'Regex': "'eval_overall_f1': (.*?),"},
            {'Name': 'eval_overall_precision', 'Regex': "'eval_overall_precision': (.*?),"},
        ]

        super().__init__(
            entry_point           = EstimatorConfig.entry_point,                                # train script
            source_dir            = str(Path(os.path.dirname(os.path.realpath(__file__))).parent 
                                        / EstimatorConfig.source_dir_path),                                 # directory which includes all the files needed for training
            output_path           = EstimatorConfig.training_jobs_uri,                       # s3 path to save the artifacts
            code_location         = EstimatorConfig.training_jobs_uri,                       # s3 path to stage the code during the training job
            instance_type         = EstimatorConfig.instance_type,                           # instances type used for the training job
            instance_count        = EstimatorConfig.instance_count,                          # the number of instances used for training
            base_job_name         = EstimatorConfig.job_name,                                # the name of the training job
            role                  = os.getenv('SAGEMAKER_ROLE'),                                       # Iam role used in training job to access AWS ressources, e.g. S3
            transformers_version  = EstimatorConfig.transformers_version,                   # the transformers version used in the training job
            pytorch_version       = EstimatorConfig.pytorch_version,                        # the pytorch_version version used in the training job
            py_version            = EstimatorConfig.py_version,                             # the python version used in the training job
            hyperparameters       = hyperparameters,                                        # the hyperparameters used for the training job
            metric_definitions    = metrics_defintions,                                     # the metrics used to track the training job
            environment           = {
                "HUGGINGFACE_HUB_CACHE": "/tmp/.cache",
                "COMET_API_KEY": os.getenv("COMET_API_KEY")
            },                                                                              # set env variables
    )


if __name__ == "__main__":

    layoutLM_estimator = LayoutLMEstimator()

    # define a data input dictonary with our uploaded s3 uris
    # SM_CHANNEL_{channel_name}
    # Need to ref the correct Channel (https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    data = {'training': ModelTrainingConfig.dataset_uri}

    # starting the train job with our uploaded datasets as input
    layoutLM_estimator.fit(data, wait=True)
