# https://github.com/aws/amazon-sagemaker-examples/blob/main/async-inference/Async-Inference-Walkthrough.ipynb
# https://github.com/huggingface/notebooks/blob/main/sagemaker/16_async_inference_hf_hub/sagemaker-notebook.ipynb
import time
import urllib
import logging

import boto3
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.async_inference.waiter_config import WaiterConfig
from sagemaker.async_inference.async_inference_response import AsyncInferenceResponse
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from config import EndpointDeploymentConfig

start = time.time()

# Temporary URLS
PRESIGNED_URLS = [
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_11.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aDGV1LWNlbnRyYWwtMSJHMEUCIQDGHd%2BAdfvdk7ALPzYVcfngLSIGjpfEr1h3RPpZz899FAIgdiUv3lP%2Fy7ILQErsNs44P5t3MCEGrZMA5U%2B8BfFH8%2B8q5AIIWBABGgwyNjU4OTA3NjE3NzciDIXklVVIqIHKcA6DaSrBAtRSQs1yZWfz6ZyuYB%2BfEjTV%2FnW76U4zH%2F52p1fFjvAA2sOp8z99PU4Y5XJke%2BAvZe3ZUckgDigF1CxP8fBg0J8uDjT1%2FHwjI9Nh3bTCEm7qdDR4Dv%2BuTBXnvVMQ0eo1Xfo64m1SMhoCE5BslZ2EdqWyk%2BIpElmyKHSIhvzPYzN67v8KJprf4JzP5uSIn6WN4qv5JrFMM0BFaWcKgJPnqSL1Ann03ZylhFRKNHaFdmV8N4SJ5MYT8%2Flcta3LzfvRI7OBXLl9tOFO%2FteTAidMVChIuDTUuYTxeQf9IdV1gd1rFOwERpwnGA%2B2J0UzIEVn%2Fdi1hI%2FEupHYzGS7qdXYpnIsICKiQlXg4mH%2FgXweB%2Baw88nkL4tPUYXw2xZwxu2%2FadOw9F7svmF54sTU1Np3ejamQLOfsXJSx8JKIjo1y2FILzDQnsWvBjqzAmcGge4fM0KChEv523BpymUW6ol4mgUGMinPhZHS1gDoYhrkYEBJ%2F4VRv%2F4bMdj0pEYy8yVpTNh0h857ukKACF33kJPGQmduF4kO9r3IA0%2FUByupWdmHOwvgSg1I0ei2bdQu5IFbd2zcbsV74dZvye%2F2smHrxSOvWe813Z1Ybom4Hu4Ok%2BTfE%2F8Vd%2FVKDqqR3H8Ec2eW1u%2Bs%2BWsLrsEIyjvisZj%2BMOy2RABAylpcW2yo0j9YEXS0eNdJzRbul3NecJyBpj2x24JUXL%2FImBJSR4wVvKJefaiHgCZYKjz32FoiEF9ITRuRshd8kRSYy1rQCwDVcnOdPpSSbsyNfBub97HOTWRQn14OtGfam%2BoliwSGVU%2BLXaUlWC0QKX%2FXmwx0hE2dmtB25hYJf6g12gAqAO%2FwRhw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240313T100545Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36000&X-Amz-Credential=ASIAT32CTBQYWK7QJZWC%2F20240313%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=82f10ea57a2d8abe26b8f37bd08d5cefaa15b61f7b4a12ea6de2ead6b0fa3e35",
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_ext_2.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aDGV1LWNlbnRyYWwtMSJHMEUCIQDGHd%2BAdfvdk7ALPzYVcfngLSIGjpfEr1h3RPpZz899FAIgdiUv3lP%2Fy7ILQErsNs44P5t3MCEGrZMA5U%2B8BfFH8%2B8q5AIIWBABGgwyNjU4OTA3NjE3NzciDIXklVVIqIHKcA6DaSrBAtRSQs1yZWfz6ZyuYB%2BfEjTV%2FnW76U4zH%2F52p1fFjvAA2sOp8z99PU4Y5XJke%2BAvZe3ZUckgDigF1CxP8fBg0J8uDjT1%2FHwjI9Nh3bTCEm7qdDR4Dv%2BuTBXnvVMQ0eo1Xfo64m1SMhoCE5BslZ2EdqWyk%2BIpElmyKHSIhvzPYzN67v8KJprf4JzP5uSIn6WN4qv5JrFMM0BFaWcKgJPnqSL1Ann03ZylhFRKNHaFdmV8N4SJ5MYT8%2Flcta3LzfvRI7OBXLl9tOFO%2FteTAidMVChIuDTUuYTxeQf9IdV1gd1rFOwERpwnGA%2B2J0UzIEVn%2Fdi1hI%2FEupHYzGS7qdXYpnIsICKiQlXg4mH%2FgXweB%2Baw88nkL4tPUYXw2xZwxu2%2FadOw9F7svmF54sTU1Np3ejamQLOfsXJSx8JKIjo1y2FILzDQnsWvBjqzAmcGge4fM0KChEv523BpymUW6ol4mgUGMinPhZHS1gDoYhrkYEBJ%2F4VRv%2F4bMdj0pEYy8yVpTNh0h857ukKACF33kJPGQmduF4kO9r3IA0%2FUByupWdmHOwvgSg1I0ei2bdQu5IFbd2zcbsV74dZvye%2F2smHrxSOvWe813Z1Ybom4Hu4Ok%2BTfE%2F8Vd%2FVKDqqR3H8Ec2eW1u%2Bs%2BWsLrsEIyjvisZj%2BMOy2RABAylpcW2yo0j9YEXS0eNdJzRbul3NecJyBpj2x24JUXL%2FImBJSR4wVvKJefaiHgCZYKjz32FoiEF9ITRuRshd8kRSYy1rQCwDVcnOdPpSSbsyNfBub97HOTWRQn14OtGfam%2BoliwSGVU%2BLXaUlWC0QKX%2FXmwx0hE2dmtB25hYJf6g12gAqAO%2FwRhw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240313T100521Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36000&X-Amz-Credential=ASIAT32CTBQYWK7QJZWC%2F20240313%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=8cc869eed276ea421eb63c47d0858068176cb8ed5d51c93037e119fac0cedc2b"
]

payload = {"urls": PRESIGNED_URLS}
# with open("async_payload.json", "w") as f:
#     json.dump(payload, f, indent=4)


predictor = Predictor(
    endpoint_name=EndpointDeploymentConfig.endpoint_name,
    sagemaker_session=Session(),
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

waiter_config = WaiterConfig(
    delay=15,
    max_attempts=60
)
predictor_async = AsyncPredictor(predictor=predictor)

# predictor_async.predict() wait for the output
# predictor_async.predictor_async() doesn't wait! 
result = predictor_async.predict_async(
    data=payload,
    input_path=EndpointDeploymentConfig.async_input_path,
    # waiter_config=waiter_config
)
print(result.get_result(waiter_config=waiter_config))

print(f"Time taken: {time.time() - start}s")

# client = boto3.client('sagemaker-runtime')

# endpoint_name = EndpointDeploymentConfig.endpoint_name
# content_type = "application/json"
# payload = payload   

# response = client.invoke_endpoint_async(
#     EndpointName=endpoint_name, 
#     ContentType=content_type,
#     InputLocation = "s3://invoice-reader-project/production/async_input/async_payload.json"
# )

# sagemaker_session = Session()

# AsyncInferenceResponse()

# output_location = response["OutputLocation"]
# def get_output(output_location):
#     output_url = urllib.parse.urlparse(output_location)
#     bucket = output_url.netloc
#     key = output_url.path[1:]
#     while True:
#         try:
#             return sagemaker_session.read_s3_file(
#                                         bucket=output_url.netloc, 
#                                         key_prefix=output_url.path[1:])
#         except ClientError as e:
#             if e.response['Error']['Code'] == 'NoSuchKey':
#                 print("waiting for output...")
#                 time.sleep(2)
#                 continue
#             raise

# output = get_output(output_location)
# print(f"Output: {output}")