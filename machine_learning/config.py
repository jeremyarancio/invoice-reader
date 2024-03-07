from dataclasses import dataclass


@dataclass
class EndpointDeploymentConfig:
  initial_instance_count = 1
  instance_type = "ml.t2.medium"


@dataclass
class EstimatorConfig:
  training_jobs_uri = "s3://invoice-reader-project/training-jobs/"
  source_dir_path = "code"
  entry_point = "train.py"
  instance_type = "ml.g4dn.xlarge"
  instance_count = 1
  job_name = "layoutLMTraining"
  transformers_version = "4.26"
  pytorch_version = "1.13"
  py_version = "py39"
  tags = "LayoutLM" # CometML experiment tags -> Need to be a string for environment variable in Training jobs
  comet_project_name = "invoice-reader"



@dataclass
class ModelTrainingConfig:
  pretrained_model_name = "microsoft/layoutlm-base-uncased"
  labels = [
    "O",
    "S-name",
    "S-adress",
    "B-name",
    "B-adress",
    "Total net",
    "Total gross"
  ]
  epochs = 20
  per_device_train_batch_size = 16
  per_device_eval_batch_size = 8
  lr = 5e-5
  warmup_steps = 10
  dataset_uri = "s3://invoice-reader-project/data/training/datasets/dataset_ocr_v1/"