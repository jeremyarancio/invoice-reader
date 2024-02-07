import logging
from pathlib import Path
from PIL import Image
from typing import List, Iterator, Mapping, Optional
from io import BytesIO
from tqdm import tqdm
import json
import os
from uuid import uuid4

from datasets import load_dataset, disable_caching
import pytesseract
from pdf2image import convert_from_path
import boto3
from botocore.exceptions import ClientError


disable_caching()

HF_DATASET_ID = "katanaml-org/invoices-donut-data-v1"
PRIVATE_INVOICES_PDF_DIR = Path(".private/invoices_pdf/")
PRIVATE_INVOICES_PNG_DIR = Path(".private/invoices_png/")
BUCKET_NAME = "invoice-reader-project"
S3_FOLDER = "data/training/documents/"
DATASET_SPLIT = "validation"
STREAMING = True
OCR_JSON_PATH = "data/invoices_ocr_data.json"

# tesseract output levels for the level of detail for the bounding boxes
LEVELS = {"page_num": 1, "block_num": 2, "par_num": 3, "line_num": 4, "word_num": 5}


def transform_pdf_into_image(pdf_dir: Path, png_dir: Path = False) -> List[Image.Image]:
    """Convert a pdf into images from a local folder."""
    images = []
    for idx, filepath in enumerate(pdf_dir.iterdir()):
        # Take the first page
        pdf_image = convert_from_path(pdf_path=filepath.as_posix())[0]
        # Save
        if png_dir:
            pdf_image.save((png_dir / f"invoice_{idx}.png").as_posix())
        images.append(pdf_image)
    return images


def extract_images_from_dataset(
    dataset_id: str, split: str, bucket_name: str, prefix: str
) -> None:
    """From huggingface datasets, extract images and store them in s3"""
    s3_client = boto3.client("s3")
    dataset = load_dataset(path=dataset_id, split=split)
    images = dataset["image"]
    for idx, image in enumerate(images):
        _upload_image_to_s3(
            s3_client, image, bucket_name, prefix + "invoice_ext_{}.png".format(idx)
        )


def _upload_image_to_s3(s3_client, image: Image, bucket_name: str, prefix: str) -> bool:
    """"""
    # Convert PIL image to bytes
    img_byte_array = BytesIO()
    image.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)
    try:
        s3_client.upload_fileobj(img_byte_array, bucket_name, prefix)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_local_images_to_s3(png_dir: Path, bucket_name: str, prefix: str) -> None:
    """"""
    s3_client = boto3.client("s3")
    for idx, image_path in enumerate(png_dir.iterdir()):
        image = Image.open(image_path)
        _upload_image_to_s3(
            s3_client, image, bucket_name, prefix + "invoice_{}.png".format(idx)
        )


def load_images_from_s3(bucket, prefix) -> Iterator[Image.Image]:
    """"""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".png"):
            file_stream = s3.get_object(Bucket=bucket, Key=key)["Body"]
            image = Image.open(file_stream)
            yield image


def prepare_data_for_ls(
    images: Iterator[Image.Image], save_path: Optional[Path] = None
) -> Mapping:
    """Parse image and extract words with bboxes for Label Studio
    """
    output = []
    for image in tqdm(images):
        tesseract_output = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        create_image_url(image.file)
       
    if save_path:
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
    return output


def create_image_url(filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f"http://localhost:8080/{filename}"


def convert_to_ls(image, tesseract_output: Mapping, per_level: str = "block_num"):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template

    * https://labelstud.io/guide/predictions.html#Import-bbox-and-choice-pre-annotations-for-images
    * https://labelstud.io/blog/improve-ocr-quality-for-receipt-processing-with-tesseract-and-label-studio/#takeaways-for-ocr-with-tesseract-and-label-studio

    """
    image_width, image_height = image.size
    per_level_idx = LEVELS[per_level]
    results = []
    all_scores = []
    for i, level_idx in enumerate(tesseract_output["level"]):
        if level_idx == per_level_idx:
            bbox = {
                "x": 100 * tesseract_output["left"][i] / image_width,
                "y": 100 * tesseract_output["top"][i] / image_height,
                "width": 100 * tesseract_output["width"][i] / image_width,
                "height": 100 * tesseract_output["height"][i] / image_height,
                "rotation": 0,
                "rectangle_labels": ["O"] # Default label
            }

            words, confidences = [], []
            for j, curr_id in enumerate(tesseract_output[per_level]):
                if curr_id != tesseract_output[per_level][i]:
                    continue
                word = tesseract_output["text"][j]
                confidence = tesseract_output["conf"][j]
                words.append(word)
                if confidence != "-1":
                    confidences.append(float(confidence / 100.0))

            text = " ".join(words).strip()
            if not text:
                continue
            region_id = str(uuid4())[:10]
            score = sum(confidences) / len(confidences) if confidences else 0
            bbox_result = {
                "id": region_id,
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectangle",
                "value": bbox,
            }
            transcription_result = {
                "id": region_id,
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea",
                "value": dict(text=[text], **bbox),
                "score": score,
            }
            results.extend([bbox_result, transcription_result])
            all_scores.append(score)

    return {
        "data": {"ocr": create_image_url(image.filename)},
        "predictions": [
            {
                "result": results,
                "score": sum(all_scores) / len(all_scores) if all_scores else 0,
            }
        ],
    }


def main():
    # images = transform_pdf_into_image(
    #     pdf_dir=PRIVATE_INVOICES_PDF_DIR,
    #     png_dir=PRIVATE_INVOICES_PNG_DIR,
    # )
    # extract_images_from_dataset(
    #     dataset_id=HF_DATASET_ID,
    #     split=DATASET_SPLIT,
    #     bucket_name=BUCKET_NAME,
    #     prefix=S3_FOLDER,
    #     streaming=STREAMING,
    # )
    # upload_local_images_to_s3(
    #     png_dir=PRIVATE_INVOICES_PNG_DIR,
    #     bucket_name=BUCKET_NAME,
    #     prefix=S3_FOLDER
    # )
    images = load_images_from_s3(bucket=BUCKET_NAME, prefix=S3_FOLDER)
    ocr_images(images=images, save_path=OCR_JSON_PATH)


if __name__ == "__main__":
    main()
