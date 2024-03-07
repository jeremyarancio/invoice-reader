import logging

from sagemaker.predictor import Predictor
from sagemaker import Session
from sagemaker.serializers import JSONSerializer

# Temporary URLS
PRESIGNED_URLS = [
    "https://invoice-reader-project.s3.eu-central-1.amazonaws.com/data/training/documents/invoice_ext_25.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJGMEQCIAQFcQ0HwxPa9PfJUm64lRI6ID0I8ZD3YK2uJeMnV85dAiBk5YGJmwktRiu281LBFuXAMwxiqGlGyTo3y0KH%2B74kcyrtAgi6%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDI2NTg5MDc2MTc3NyIMTvltw0wEEIRGE131KsECo1oJ5eF%2FO5Od9avst5potWIVM7i%2FFSfLnsWzOEErhHybd%2B0YEFCp5o%2Fiy%2BwwWWAvp7%2Fv9Agy1gcviJeFLsDE1Mva7F10UglDon7EICXpygzSW92wH4hmy%2BQRODzLcQ2GkL9Lye78GIRA4W8roF8uKIL5OCS3MwvZkKxLKr6MS%2FjuEnRdOUB5YwIHHexoNvAAYMWwsRchIAvlOqESlKdfhdteJIucR1wEyE%2F0iUsf7c6BsmLVG8avG0OTNIUbMHHCiukPoQnmTnbo9fxyhd9XerWW2OfQ5Y1HSXunOci0xO%2FhKBaQWs6%2B0tAsOiBgo991A3dUsVpWFcS0b0wLkI8q1FZrvomp6dcULBXoF6g0NbEdRSKEBrrSXPWIuJwqx4eJi9ijOMr4B0HrYkRAw8E67JRnXa%2FxEK11wK4Zeaija3HIMLX%2Bpa8GOrQCQUuDOy3vdMUKKDUK595HgQ2pqzTlHTJ5LfoKan57B6F72hJyBEfWRB1oPSvklR9Q%2FcHeIC2i0cB2TwikJFfH6Kfd0PJiMojFNbpwRZQnZ2UrOnZ2KlSsQGGX8zWomDDX1HsE9NGyZDnt3gkdngd9GS2lbvmPpdGYZgmsmeGBrZevzdDEBo%2Bji7tqNU2r%2FiUHaEHrvKpvNcTroqN5HzcUTlp130v2Xc1iLYHfho%2BVD8oStAM8c9mcVv%2BUGwkV2k6%2BCVKojk6wXEXKUwHL56QYZRhA5mcjxWOC7UZFZeau7x7xexNciqJdK4HYywteoHR%2BZO%2FN7G0qRYO4gnB4ip0%2BQQ2dDX%2BYfyPsLRAhIe8rokcS0tPT3D9LnruBEme793tA6ynm9mn0V%2FYR4rexUYeS6NqpktY%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240307T124738Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIAT32CTBQY3QRTAEAF%2F20240307%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=d2ce26bb8a880c9098c55b444ca49ac2be3e4df160acfd25f8a6ef69dd47e525"
]

payload = {"urls": PRESIGNED_URLS}

predictor = Predictor(
    endpoint_name="huggingface-pytorch-inference-2024-03-07-12-42-08-917",
    sagemaker_session=Session(),
    serializer=JSONSerializer()
)

print(predictor.predict(data=payload))