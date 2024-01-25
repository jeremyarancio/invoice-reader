import base64
from io import BytesIO
from PIL import Image

from sagemaker.huggingface import HuggingFacePredictor
from sagemaker import Session
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer

from datasets import load_dataset, disable_caching

disable_caching()

def convert_to_bytes(image: Image):
    format = image.format if image.format else 'JPEG'
    # BytesIO is a file-like buffer stored in memory
    img_byte_arr = BytesIO()
    # image.save expects a file-like as a argument
    image.save(img_byte_arr, format=format)
    # Turn the BytesIO object back into a bytes object
    return img_byte_arr.getvalue()

# create a serializer for the data
image_serializer = DataSerializer(content_type='image/x-image') # using x-image to support multiple image formats

predictor = HuggingFacePredictor(
    endpoint_name="huggingface-pytorch-inference-2024-01-25-18-24-08-402",
    sagemaker_session=Session(),
    serializer=image_serializer
)

image = Image.open(".private/invoice.png")
prediction = predictor.predict(data=convert_to_bytes(image))
print(prediction)