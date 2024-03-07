import logging
import os

from sagemaker.huggingface import HuggingFaceModel

from config import HFModelConfig, EndpointDeploymentConfig


LOGGER = logging.getLogger(__name__)
ROLE = os.getenv("SAGEMAKER_ROLE")

# Hub model configuration <https://huggingface.co/models>
HUB = {
  'HF_MODEL_ID': HFModelConfig.hf_model_id,  # model_id from hf.co/models
  'HF_TASK': HFModelConfig.hf_task           # NLP task you want to use for predictions
}


class Deployement:

    def __init__(
        self,
        initial_instance_count=EndpointDeploymentConfig.initial_instance_count,
        instance_type=EndpointDeploymentConfig.instance_type
    ) -> None:
        self.initial_instance_count = initial_instance_count
        self.instance_type = instance_type
        self.model = HuggingFaceModel(
            env=HUB,                                                # configuration for loading model from Hub
            role=ROLE,                                              # IAM role with permissions to create an endpoint
            transformers_version="4.28",                             # Transformers version used
            pytorch_version="2.0",                                  # PyTorch version used
            py_version='py310',                                      # Python version used
        )

    def deploy(self) -> None:
        """"""
        self.model.deploy(
            initial_instance_count=self.initial_instance_count,
            instance_type=self.instance_type
        )


if __name__ == "__main__":
    deployment = Deployement()
    deployment.deploy()
