import logging
import os

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serializers import JSONSerializer

from config import EndpointDeploymentConfig


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ROLE = os.getenv("SAGEMAKER_ROLE")
SOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../code")


# # Specify MemorySizeInMB and MaxConcurrency in the serverless config object
# serverless_inference_config = ServerlessInferenceConfig(
#     memory_size_in_mb=EndpointDeploymentConfig.memory_size_in_mb, 
#     max_concurrency=EndpointDeploymentConfig.max_concurrency,
# )

# create async endpoint configuration
# async_config = AsyncInferenceConfig(
#     output_path=EndpointDeploymentConfig.async_output_path,
#     # notification_config={
#             #   "SuccessTopic": "arn:aws:sns:us-east-2:123456789012:MyTopic",
#             #   "ErrorTopic": "arn:aws:sns:us-east-2:123456789012:MyTopic",
#     # }, #  Notification configuration
#     max_concurrent_invocations_per_instance=EndpointDeploymentConfig.max_concurrent_invocations_per_instance,
# )

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
    
    # model = LayoutLMModel().deploy(
    #     instance_type=EndpointDeploymentConfig.instance_type,
    #     initial_instance_count=EndpointDeploymentConfig.initial_instance_count,
    #     endpoint_name=EndpointDeploymentConfig.endpoint_name,
    #     async_inference_config=async_config
    # )

    LayoutLMModel().deploy(
        endpoint_name=EndpointDeploymentConfig.endpoint_name,
        instance_type=EndpointDeploymentConfig.instance_type,
        initial_instance_count=EndpointDeploymentConfig.initial_instance_count
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