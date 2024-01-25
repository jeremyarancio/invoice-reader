from dataclasses import dataclass


@dataclass
class HFModelConfig:
  hf_model_id = "impira/layoutlm-invoices"
  hf_task = "document-question-answering"

class EndpointDeploymentConfig:
  initial_instance_count = 1
  instance_type = "ml.t2.medium"