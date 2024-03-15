from dataclasses import dataclass


@dataclass
class EndpointDeploymentConfig:
  endpoint_name = "layoutlm-invoice-v1"
  initial_instance_count = 1
  instance_type = "ml.c5.xlarge"
  py_version = "py39"
  pytorch_version = "1.13"
  transformers_version="4.26"
  entry_point = "inference.py"
  model_data = "s3://invoice-reader-project/training-jobs/layoutLMTraining-2024-03-04-09-40-39-507/output/model.tar.gz" # S3 artifact
  # Serverless config
  memory_size_in_mb = 5120
  max_concurrency = 5
  # Async config
  max_concurrent_invocations_per_instance = 4
  async_output_path = "s3://invoice-reader-project/production/async_output/"
  async_input_path = "s3://invoice-reader-project/production/async_input/async_payload.json"

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