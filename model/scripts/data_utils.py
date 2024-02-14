import logging
from pathlib import Path
from PIL import Image
from typing import List, Iterator, Mapping, Optional, Tuple
from io import BytesIO
from tqdm import tqdm
import json
from uuid import uuid4

from datasets import load_dataset, disable_caching, Dataset, DatasetDict
import pytesseract
from pdf2image import convert_from_path
import boto3
from botocore.exceptions import ClientError
import numpy as np


disable_caching()

HF_DATASET_ID = "katanaml-org/invoices-donut-data-v1"
PRIVATE_INVOICES_PDF_DIR = Path(".private/invoices_pdf/")
PRIVATE_INVOICES_PNG_DIR = Path(".private/invoices_png/")
BUCKET_NAME = "invoice-reader-project"
S3_DOC_PREFIX = "data/training/documents/"
S3_LS_DATA_KEY = (
    "data/training/label-studio/project-5-at-2024-02-13-15-29-56d382e2.json"
)
S3_DATASET_KEY = "data/training/datasets/dataset_ocr_v1"
DATASET_SPLIT = "validation"
STREAMING = True
OCR_JSON_PATH = "data/invoices_ocr_data.json"


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
            # Add the image presigned-url
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=21600,
            )
            image.filename = url
            yield image


def prepare_data_for_ls(
    images: Iterator[Image.Image], save_path: Optional[Path] = None
) -> List[Mapping]:
    """Parse image and extract words with bboxes for Label Studio."""
    tasks = []
    for image in tqdm(images):
        tesseract_output = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        task = convert_to_ls(image, tesseract_output)
        tasks.append(task)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(tasks, f, indent=2)
    return tasks


def create_image_url(filename):
    """
    The image data in this example task references an uploaded file,
    identified by the source_filename assigned by Label Studio after uploading the image.
    The best way to reference image data is using presigned URLs for images stored in cloud storage, or absolute paths
    to image data stored in local storage and added to Label Studio by syncing storage.
    """
    return NotImplementedError


def convert_to_ls(
    image: Image.Image, tesseract_output: Mapping, default_label: str = "O"
):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template

    * https://labelstud.io/guide/predictions.html#Import-bbox-and-choice-pre-annotations-for-images
    * https://labelstud.io/blog/improve-ocr-quality-for-receipt-processing-with-tesseract-and-label-studio/#takeaways-for-ocr-with-tesseract-and-label-studio

    """
    image_width, image_height = image.size
    results = []
    all_scores = []
    for i, text in enumerate(tesseract_output["text"]):
        if text.strip() != "" and tesseract_output["conf"][i] != "-1":
            bbox = {
                "x": 100 * tesseract_output["left"][i] / image_width,
                "y": 100 * tesseract_output["top"][i] / image_height,
                "width": 100 * tesseract_output["width"][i] / image_width,
                "height": 100 * tesseract_output["height"][i] / image_height,
                "rotation": 0,
            }
            region_id = str(uuid4())[:10]
            score = tesseract_output["conf"][i]
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
            label_result = {
                "id": region_id,
                "from_name": "label",
                "to_name": "image",
                "type": "labels",
                "value": dict(labels=[default_label], **bbox),
            }
            results.extend([bbox_result, label_result, transcription_result])
            all_scores.append(score)

    return {
        "data": {"ocr": image.filename},  # Presigned URL
        "predictions": [
            {
                "result": results,
                "score": sum(all_scores) / len(all_scores) if all_scores else 0,
            }
        ],
    }


def unormalize_bbox_from_ls(
    bboxes: List[Tuple[int, int, int, int]], 
    width: int, 
    height: int
) -> List[List]:
    """To label documents, each bbox was normalized into 100*x0/W."""
    x = np.array(bboxes)
    v = (width/100, height/100, width/100, height/100)
    X = x*v
    return X.round(2).tolist()
    


def ls_to_dataset(ls_data: List[Mapping], test_size: float = 0.3) -> DatasetDict:
    """"""
    mapping = []
    for idx, doc in enumerate(ls_data):
        original_width = doc["bbox"][0]["original_width"]
        original_height = doc["bbox"][0]["original_height"]
        bboxes = [
            (
                bbox["x"], 
                bbox["y"], 
                bbox["x"] + bbox["width"], 
                bbox["y"] + bbox["height"]
            ) for bbox in doc["bbox"]
        ]
        bboxes = unormalize_bbox_from_ls(
            bboxes=bboxes,
            width=original_width,
            height=original_height
        )
        labels = [label["labels"][0] for label in doc["label"]]
        words = [word for word in doc["transcription"]]
        mapping.append(
            {
                "doc_id": idx,
                "bboxes": bboxes,
                "words": words,
                "labels": labels,
                "original_width": original_width,
                "orginal_height": original_height
            }
        )
    dataset_dict = Dataset.from_list(mapping).train_test_split(test_size=test_size, shuffle=True)
    return dataset_dict


def main():
    # images = transform_pdf_into_image(
    #     pdf_dir=PRIVATE_INVOICES_PDF_DIR,
    #     png_dir=PRIVATE_INVOICES_PNG_DIR,
    # )
    # extract_images_from_dataset(
    #     dataset_id=HF_DATASET_ID,
    #     split=DATASET_SPLIT,
    #     bucket_name=BUCKET_NAME,
    #     prefix=S3_DOC_PREFIX,
    #     streaming=STREAMING,
    # )
    # upload_local_images_to_s3(
    #     png_dir=PRIVATE_INVOICES_PNG_DIR,
    #     bucket_name=BUCKET_NAME,
    #     prefix=S3_DOC_PREFIX
    # )
    # images = load_images_from_s3(bucket=BUCKET_NAME, prefix=S3_DOC_PREFIX)
    # prepare_data_for_ls(images=images, save_path="data/invoices_ocr_data.json")
    s3 = boto3.client("s3")
    ls_data = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=S3_LS_DATA_KEY)["Body"].read().decode("utf-8"))
    dataset = ls_to_dataset(ls_data)
    dataset.save_to_disk(f"s3://{BUCKET_NAME}/{S3_DATASET_KEY}")
    # s3.upload_fileobj(dataset, Bucket=BUCKET_NAME, Key=S3_DATASET_KEY)


if __name__ == "__main__":
    main()
