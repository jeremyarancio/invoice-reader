provider "aws" {
  region = "eu-central-1"
}


variable "endpoint_name" {
  type    = string
  default = "layoutlm-invoice-v1"
}


# +-------------------------------------------------------
# IAM
# +-------------------------------------------------------

# Define the IAM role for the Lambda function
resource "aws_iam_role" "lambda_role" {
  name = "lambda-sagemaker-role" # Update with your desired IAM role name
  assume_role_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : {
          "Service" : "lambda.amazonaws.com"
        },
        "Action" : "sts:AssumeRole"
      },
    ]
  })
}

# Define the IAM policy for SageMaker endpoint invocation
resource "aws_iam_policy" "lambda_sagemaker_policy" {
  name        = "lambda-sagemaker-policy" # Update with your desired IAM policy name
  description = "Policy for Lambda to invoke SageMaker endpoint"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "sagemaker:InvokeEndpoint",
        Resource = "*",
      },
    ],
  })
}

# Attach policies to the IAM role
resource "aws_iam_role_policy_attachment" "attach_iam_policy_to_iam_role" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_sagemaker_policy.arn
}


# +-------------------------------------------------------
# Function
# +-------------------------------------------------------


data "archive_file" "lambda" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/invoke_endpoint.py"
  output_path = "${path.module}/../lambda_functions/zip/invoke_endpoint.zip"
}

resource "aws_lambda_function" "terraform_lambda_func" {
  filename      = "${path.module}/../lambda_functions/zip/invoke_endpoint.zip"
  function_name = "InvokeInvoiceReader"
  role          = aws_iam_role.lambda_role.arn
  handler       = "invoke_endpoint.lambda_handler"
  runtime       = "python3.9"
  depends_on    = [aws_iam_role_policy_attachment.attach_iam_policy_to_iam_role]
  timeout = 30
  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = var.endpoint_name
    }
  }
}

