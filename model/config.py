from dataclasses import dataclass


@dataclass
class HFModelConfig:
  hf_model_id = "impira/layoutlm-invoices"
  hf_task = "document-question-answering"


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
  epochs = 1
  per_device_train_batch_size = 16
  per_device_eval_batch_size = 8
  lr = 3e-5
  dataset_uri = "s3://invoice-reader-project/data/training/datasets/dataset_ocr_v1/"