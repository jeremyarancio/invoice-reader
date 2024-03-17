provider "aws" {
    region = "eu-central-1"
}

# +-----------------------------------------------------------------
# Variables 
# +-----------------------------------------------------------------
variable "sagemaker_execution_role" {
  type = string
  description = "Sagemaker arn stored as an environment variable."
}

variable "image_uri" {
  type = string
  default = "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"
  description = "Check the offfcial AWS repo: https://github.com/aws/deep-learning-containers/blob/master/available_images.md"
}

variable "endpoint_name" {
  type = string
  default = "layoutlm-invoice-v1"
}


# +-----------------------------------------------------------------
# Outputs 
# +-----------------------------------------------------------------

output "execution_role_arn" {
  value = var.sagemaker_execution_role
}

# +-----------------------------------------------------------------
# Resources 
# +-----------------------------------------------------------------

resource "aws_sagemaker_model" "sagemaker_model" {
  name = var.endpoint_name
  execution_role_arn = var.sagemaker_execution_role
  primary_container {
    image = var.image_uri
    mode = "SingleModel"
    model_data_url = "s3://invoice-reader-project/training-jobs/layoutLMTraining-2024-03-04-09-40-39-507/output/model.tar.gz"
  }
  tags = {
    Name = "sagemaker-model-terraform"
  }
}

resource "aws_sagemaker_endpoint_configuration" "sagemaker_endpoint_configuration" {
  name = var.endpoint_name
  
  production_variants {
    initial_instance_count = 1
    instance_type = "ml.c5.xlarge"
    model_name = aws_sagemaker_model.sagemaker_model.name
    variant_name = "AllTraffic"
    container_startup_health_check_timeout_in_seconds = 300
  }
  tags = {
    Name = "terraform"
  }
}

resource "aws_sagemaker_endpoint" "sagemaker_endpoint" {
  name = var.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.sagemaker_endpoint_configuration.name  
  tags = {
    Name = "terraform"
  }
}
