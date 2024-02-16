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
  instance_type = "ml.t2.medium"
  instance_count = 1
  job_name = "training"
  epochs = 15
  per_device_train_batch_size = 16
  per_device_ebal_batch_size = 8
  evaluation_strategy = "epoch"
  save_strategy = "epoch"
  fp16 = True
  lr = 3e-5
  gradient_checkpointing = False
  load_best_model_at_end = True
  metric_for_best_model = "overall_f1"
  push_to_hub = False

@dataclass
class ModelConfig:
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