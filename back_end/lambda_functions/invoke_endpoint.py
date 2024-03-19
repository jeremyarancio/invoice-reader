import json
import boto3
import logging


logger = logging.getLogger()
logger.setLevel("INFO")

client = boto3.client('sagemaker-runtime')
endpoint_name = "layoutlm-invoice-v1"
content_type = "application/json"


def lambda_handler(event, context):
    
    logging.info("Received event: " + json.dumps(event, indent=2))
    payload = event.pop("body", None) # "body" value is json using requests python library. We keep it like that since it is load into invoke_endpoint()
    if payload:
        # Requests dictionnary containing {'body': json_object}
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType=content_type,
            Body=payload
        )
    else:
        # Deserialized JSON event
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType=content_type,
            Body=json.dumps(event)
        )
        
    predictions = response["Body"].read().decode()
    
    return {
        'statusCode': 200,
        'body': json.dumps(predictions)
    }
