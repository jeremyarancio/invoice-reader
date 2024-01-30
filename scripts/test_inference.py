import base64
from io import BytesIO
from PIL import Image

from sagemaker.huggingface import HuggingFacePredictor
from sagemaker import Session


def convert_to_bytes(image: Image) -> str:
    image = Image.open(".private/invoice.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# create a serializer for the data
# image_serializer = IdentitySerializer() # using x-image to support multiple image formats

predictor = HuggingFacePredictor(
    endpoint_name="huggingface-pytorch-inference-2024-01-30-17-40-07-790",
    sagemaker_session=Session(),
    # serializer=IdentitySerializer()
)

image = Image.open(".private/invoice.png")
prediction = predictor.predict(
    {
        "image": "https://www.invoicesimple.com/wp-content/uploads/2018/06/Sample-Invoice-printable.png",
        "question": "What is the total gross amount?",
    }
)
print(prediction)

# ERROR: "message": "If you provide an image without word_boxes, then the pipeline will run OCR using Tesseract, but pytesseract is not available"