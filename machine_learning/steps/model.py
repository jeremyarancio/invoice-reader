import logging
import os

from sagemaker.huggingface import HuggingFaceModel

from config import EndpointDeploymentConfig


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ROLE = os.getenv("SAGEMAKER_ROLE")
SOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../code")


class LayoutLMModel(HuggingFaceModel):
    def __init__(
        self,
        role=ROLE,
        model_data=EndpointDeploymentConfig.model_data,
        entry_point=os.path.join(SOURCE_DIR, EndpointDeploymentConfig.entry_point),                                                                                               # IAM role with permissions to create an endpoint
        transformers_version=EndpointDeploymentConfig.transformers_version,    # Transformers version used
        pytorch_version=EndpointDeploymentConfig.pytorch_version,              # PyTorch version used
        py_version=EndpointDeploymentConfig.py_version,                        # Python version used,
    ) -> None:
        super().__init__(
            model_data=model_data,
            role=role,
            entry_point=entry_point,
            transformers_version=transformers_version,    # Transformers version used
            pytorch_version=pytorch_version,              # PyTorch version used
            py_version=py_version                        # Python version used
        )


if __name__ == "__main__":
    model = LayoutLMModel().deploy(
        initial_instance_count=EndpointDeploymentConfig.initial_instance_count,
        instance_type=EndpointDeploymentConfig.instance_type,
        volume_size=30
    )


    # model = HuggingFaceModel(
    #     role=ROLE,
    #     model_data=EndpointDeploymentConfig.model_data,
    #     entry_point=os.path.join(SOURCE_DIR, EndpointDeploymentConfig.entry_point),                                                                                               # IAM role with permissions to create an endpoint
    #     transformers_version=EndpointDeploymentConfig.transformers_version,    # Transformers version used
    #     pytorch_version=EndpointDeploymentConfig.pytorch_version,              # PyTorch version used
    #     py_version=EndpointDeploymentConfig.py_version,                        # Python version used,
    # )

    # model.deploy(
    #     initial_instance_count=EndpointDeploymentConfig.initial_instance_count,
    #     instance_type=EndpointDeploymentConfig.instance_type
    # )