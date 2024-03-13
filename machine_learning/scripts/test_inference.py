import logging

from sagemaker.predictor import Predictor
from sagemaker import Session
from sagemaker.serializers import JSONSerializer

from config import EndpointDeploymentConfig

# Temporary URLS
PRESIGNED_URLS = [
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_ext_10.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aDGV1LWNlbnRyYWwtMSJHMEUCIQDGHd%2BAdfvdk7ALPzYVcfngLSIGjpfEr1h3RPpZz899FAIgdiUv3lP%2Fy7ILQErsNs44P5t3MCEGrZMA5U%2B8BfFH8%2B8q5AIIWBABGgwyNjU4OTA3NjE3NzciDIXklVVIqIHKcA6DaSrBAtRSQs1yZWfz6ZyuYB%2BfEjTV%2FnW76U4zH%2F52p1fFjvAA2sOp8z99PU4Y5XJke%2BAvZe3ZUckgDigF1CxP8fBg0J8uDjT1%2FHwjI9Nh3bTCEm7qdDR4Dv%2BuTBXnvVMQ0eo1Xfo64m1SMhoCE5BslZ2EdqWyk%2BIpElmyKHSIhvzPYzN67v8KJprf4JzP5uSIn6WN4qv5JrFMM0BFaWcKgJPnqSL1Ann03ZylhFRKNHaFdmV8N4SJ5MYT8%2Flcta3LzfvRI7OBXLl9tOFO%2FteTAidMVChIuDTUuYTxeQf9IdV1gd1rFOwERpwnGA%2B2J0UzIEVn%2Fdi1hI%2FEupHYzGS7qdXYpnIsICKiQlXg4mH%2FgXweB%2Baw88nkL4tPUYXw2xZwxu2%2FadOw9F7svmF54sTU1Np3ejamQLOfsXJSx8JKIjo1y2FILzDQnsWvBjqzAmcGge4fM0KChEv523BpymUW6ol4mgUGMinPhZHS1gDoYhrkYEBJ%2F4VRv%2F4bMdj0pEYy8yVpTNh0h857ukKACF33kJPGQmduF4kO9r3IA0%2FUByupWdmHOwvgSg1I0ei2bdQu5IFbd2zcbsV74dZvye%2F2smHrxSOvWe813Z1Ybom4Hu4Ok%2BTfE%2F8Vd%2FVKDqqR3H8Ec2eW1u%2Bs%2BWsLrsEIyjvisZj%2BMOy2RABAylpcW2yo0j9YEXS0eNdJzRbul3NecJyBpj2x24JUXL%2FImBJSR4wVvKJefaiHgCZYKjz32FoiEF9ITRuRshd8kRSYy1rQCwDVcnOdPpSSbsyNfBub97HOTWRQn14OtGfam%2BoliwSGVU%2BLXaUlWC0QKX%2FXmwx0hE2dmtB25hYJf6g12gAqAO%2FwRhw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240313T075041Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIAT32CTBQYWK7QJZWC%2F20240313%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=5877dd9d2655af8c86a3b6eafe046e1fb134e190ceebc3db2192384d5e39c9e8",
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_15.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aDGV1LWNlbnRyYWwtMSJHMEUCIQDGHd%2BAdfvdk7ALPzYVcfngLSIGjpfEr1h3RPpZz899FAIgdiUv3lP%2Fy7ILQErsNs44P5t3MCEGrZMA5U%2B8BfFH8%2B8q5AIIWBABGgwyNjU4OTA3NjE3NzciDIXklVVIqIHKcA6DaSrBAtRSQs1yZWfz6ZyuYB%2BfEjTV%2FnW76U4zH%2F52p1fFjvAA2sOp8z99PU4Y5XJke%2BAvZe3ZUckgDigF1CxP8fBg0J8uDjT1%2FHwjI9Nh3bTCEm7qdDR4Dv%2BuTBXnvVMQ0eo1Xfo64m1SMhoCE5BslZ2EdqWyk%2BIpElmyKHSIhvzPYzN67v8KJprf4JzP5uSIn6WN4qv5JrFMM0BFaWcKgJPnqSL1Ann03ZylhFRKNHaFdmV8N4SJ5MYT8%2Flcta3LzfvRI7OBXLl9tOFO%2FteTAidMVChIuDTUuYTxeQf9IdV1gd1rFOwERpwnGA%2B2J0UzIEVn%2Fdi1hI%2FEupHYzGS7qdXYpnIsICKiQlXg4mH%2FgXweB%2Baw88nkL4tPUYXw2xZwxu2%2FadOw9F7svmF54sTU1Np3ejamQLOfsXJSx8JKIjo1y2FILzDQnsWvBjqzAmcGge4fM0KChEv523BpymUW6ol4mgUGMinPhZHS1gDoYhrkYEBJ%2F4VRv%2F4bMdj0pEYy8yVpTNh0h857ukKACF33kJPGQmduF4kO9r3IA0%2FUByupWdmHOwvgSg1I0ei2bdQu5IFbd2zcbsV74dZvye%2F2smHrxSOvWe813Z1Ybom4Hu4Ok%2BTfE%2F8Vd%2FVKDqqR3H8Ec2eW1u%2Bs%2BWsLrsEIyjvisZj%2BMOy2RABAylpcW2yo0j9YEXS0eNdJzRbul3NecJyBpj2x24JUXL%2FImBJSR4wVvKJefaiHgCZYKjz32FoiEF9ITRuRshd8kRSYy1rQCwDVcnOdPpSSbsyNfBub97HOTWRQn14OtGfam%2BoliwSGVU%2BLXaUlWC0QKX%2FXmwx0hE2dmtB25hYJf6g12gAqAO%2FwRhw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240313T075112Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIAT32CTBQYWK7QJZWC%2F20240313%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=b0cf2e2901c7186ff380d6679d79ef0c8d400836b02123004e8f2fb74ae55cdd"
]

payload = {"urls": PRESIGNED_URLS}

# predictor = Predictor(
#     endpoint_name=EndpointDeploymentConfig.endpoint_name,
#     sagemaker_session=Session(),
#     serializer=JSONSerializer()
# )

# print(predictor.predict(data=payload))


# With Invoke_endpoint
import boto3
import json

client = boto3.client('sagemaker-runtime')

endpoint_name = EndpointDeploymentConfig.endpoint_name
content_type = "application/json"
payload = payload   

response = client.invoke_endpoint(
    EndpointName=endpoint_name, 
    ContentType=content_type,
    Body=json.dumps(payload)
    )

print(response["Body"].read().decode())               