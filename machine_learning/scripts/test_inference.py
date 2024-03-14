import logging
import json
import urllib
import time

from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.async_inference.waiter_config import WaiterConfig
from sagemaker import Session
from sagemaker.serializers import JSONSerializer
import boto3
from botocore.exceptions import ClientError

from config import EndpointDeploymentConfig

# Temporary URLS
PRESIGNED_URLS = [
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_ext_25.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGkaDGV1LWNlbnRyYWwtMSJGMEQCIGLvmyJHYvjzWuWsWu13ieE1k%2FR4k65H%2B96%2BXfLfV3i%2BAiAFYhcwZKdQ9XDcRWwsPfxAdO5WIcnXJAF%2Bc%2BpkEieXmyrkAghyEAEaDDI2NTg5MDc2MTc3NyIMbNKQmmBPKMP3QbYRKsECKMPRrXryNTCaUHWYQVpFUDOLIUPO4F9XpKBKuoM03OtVBafNsw%2BVxEbmZ7ML0IRjR482deJLj%2Fgk%2BpDP0P235CGbC4jHLEHj1tdnOqgT1dF1h6PEeZo49YTkJo3ZIvgOeLueGw3%2FirPtFU0KhZZZD3R0w1PF2FTr6Liz08lAlAsHDtj%2Fe%2FEKlagAtKAChQUCSKmyZmmfic41czvwKaEpWL2Gntm5hvLBZ%2FuysFTIickjRVxi3l83PzLuAb5HR8CLjZoQKm5lj%2B6ky4LSzXlzGB8wvgV5H1B0zrKl91KQfSvOqZjNaA4AX83c6zeybxmC2S1A9KcNMud1QJS6mGCIzKtmPP42te%2Fcsl1x6ayodmgsv4FgGFjL%2BWv7ZXCa%2FkilRTbhXN8UaSC%2FsTliIwSACLSEFKlQC9SsDfPGvFI1LS6aML33yq8GOrQCJdLzTnWmHZoQ8R2X5leNi5JKnqZ7ONdVNDD7kxwgGWPRTAXp7qVYvla94Cf7lkOSOiVlbCrghehxfEp5kQ%2Fc9LNyn7whq9HkEh7arrogW1rW5KStkcSf2pmeGeCaL5KB0B4T%2BkoeG48cyRbL%2BFgsp3rm%2FbJPc6XZZfRPP0eeUPNKJ%2FYj2bd778WMxUPY2q6dFSSA1%2BMB9IFY%2BoZy%2Fp3yzxsETAay66ehLM0rEoUcOWDJmozmdoo9phg8tQln2D5kOgrVpWdi5zoluKXI5ER626GlSsrMg5ELvXWLTZUEJPRrVQyRNpEN9zBfbctqETtaBt94ArzLDmKXEf8neGKrKAjh7CaeXDP0PRCYd9%2BUCWxdVMfk4aD5UGPJRWlgIKbXNUmSvJEsu9rV5sUtcenEQy83Vi4%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240314T110057Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36000&X-Amz-Credential=ASIAT32CTBQY5OSH4HMU%2F20240314%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=5814a26b834a46ab6170e72582df87dc82c078b3d67281c2db193fcc48615850",
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_14.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGkaDGV1LWNlbnRyYWwtMSJGMEQCIGLvmyJHYvjzWuWsWu13ieE1k%2FR4k65H%2B96%2BXfLfV3i%2BAiAFYhcwZKdQ9XDcRWwsPfxAdO5WIcnXJAF%2Bc%2BpkEieXmyrkAghyEAEaDDI2NTg5MDc2MTc3NyIMbNKQmmBPKMP3QbYRKsECKMPRrXryNTCaUHWYQVpFUDOLIUPO4F9XpKBKuoM03OtVBafNsw%2BVxEbmZ7ML0IRjR482deJLj%2Fgk%2BpDP0P235CGbC4jHLEHj1tdnOqgT1dF1h6PEeZo49YTkJo3ZIvgOeLueGw3%2FirPtFU0KhZZZD3R0w1PF2FTr6Liz08lAlAsHDtj%2Fe%2FEKlagAtKAChQUCSKmyZmmfic41czvwKaEpWL2Gntm5hvLBZ%2FuysFTIickjRVxi3l83PzLuAb5HR8CLjZoQKm5lj%2B6ky4LSzXlzGB8wvgV5H1B0zrKl91KQfSvOqZjNaA4AX83c6zeybxmC2S1A9KcNMud1QJS6mGCIzKtmPP42te%2Fcsl1x6ayodmgsv4FgGFjL%2BWv7ZXCa%2FkilRTbhXN8UaSC%2FsTliIwSACLSEFKlQC9SsDfPGvFI1LS6aML33yq8GOrQCJdLzTnWmHZoQ8R2X5leNi5JKnqZ7ONdVNDD7kxwgGWPRTAXp7qVYvla94Cf7lkOSOiVlbCrghehxfEp5kQ%2Fc9LNyn7whq9HkEh7arrogW1rW5KStkcSf2pmeGeCaL5KB0B4T%2BkoeG48cyRbL%2BFgsp3rm%2FbJPc6XZZfRPP0eeUPNKJ%2FYj2bd778WMxUPY2q6dFSSA1%2BMB9IFY%2BoZy%2Fp3yzxsETAay66ehLM0rEoUcOWDJmozmdoo9phg8tQln2D5kOgrVpWdi5zoluKXI5ER626GlSsrMg5ELvXWLTZUEJPRrVQyRNpEN9zBfbctqETtaBt94ArzLDmKXEf8neGKrKAjh7CaeXDP0PRCYd9%2BUCWxdVMfk4aD5UGPJRWlgIKbXNUmSvJEsu9rV5sUtcenEQy83Vi4%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240314T110119Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36000&X-Amz-Credential=ASIAT32CTBQY5OSH4HMU%2F20240314%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=cd8492dbd79892bffcb518810942206038f046b6f824cb807661803c0b0203ad"
]

payload = {"urls": PRESIGNED_URLS}
# with open("async_payload.json", "w") as f:
#     json.dump(payload, f, indent=4)

predictor = Predictor(
    endpoint_name=EndpointDeploymentConfig.endpoint_name,
    sagemaker_session=Session(),
    serializer=JSONSerializer()
)


print(predictor.predict(data=payload))


# With Invoke_endpoint
# response = client.invoke_endpoint(
#     EndpointName=endpoint_name, 
#     ContentType=content_type,
#     Body=json.dumps(payload)
#     )

# print(response["Body"].read().decode())               


# Async
