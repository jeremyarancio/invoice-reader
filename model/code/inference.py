import io
import logging
import sys
import json
from PIL import Image
import requests

from transformers import LayoutLMForTokenClassification, AutoProcessor
import torch


logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],  # Necessary the catch training logging during training job²
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def model_fn(model_dir):
    logging.info("Start loading model.")
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv2-base-uncased", apply_ocr=True
    )
    model = LayoutLMForTokenClassification.from_pretrained(model_dir)
    logging.info("Model and tokenizer loaded.")
    return processor, model


def input_fn(input_data, content_type):
    if content_type == "application/json":
        data = json.loads(input_data)
        url = data.pop("url", None)
        image = get_image_from_url(url)
        return image
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))


def predict_fn(image, model):
    processor, model = model
    encoding = processor(
        image, return_tensors="pt", padding="max_length", truncation=True
    )
    image = encoding.pop("image")
    outputs = model(**encoding)
    results = process_outputs(
        outputs, encoding=encoding, image=image, model=model, processor=processor
    )
    return results


def get_image_from_url(url: str) -> Image:
    response = requests.get(url)
    f = io.BytesIO(response.content)
    return Image.open(f).convert("RGB")


# Handle by processor
# def normalize_bbox(bbox, width, height):
#     return [
#         int(1000 * (bbox[0] / width)),
#         int(1000 * (bbox[1] / height)),
#         int(1000 * (bbox[2] / width)),
#         int(1000 * (bbox[3] / height)),
#     ]


def unnormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]


def _process_outputs(encoding, tokenizer, labels, scores, image_size):
    width, height = image_size
    results = []
    for batch_idx, input_ids in enumerate(encoding["input_ids"]):
        entities = []
        previous_word_idx = 0
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        word_ids = encoding.word_ids(batch_index=batch_idx)
        word = ""
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif previous_word_idx != word_idx:
                # +1 because of [CLS] token
                entities.append(
                    {
                        "word": word,
                        "label": labels[batch_idx][previous_word_idx + 1],
                        "score": scores[batch_idx][previous_word_idx + 1],
                        "bbox": unnormalize_box(
                            encoding["bbox"][batch_idx][previous_word_idx + 1],
                            width=width,
                            height=height,
                        ),
                    }
                )
                word = tokens[i]
            else:
                word += tokens[i]
            previous_word_idx = word_idx
        for entity in entities:
            entity["word"] = entity["word"].replace("##", "")
        results.append(entity)
    return results


def process_outputs(outputs, encoding, image, model, processor):
    scores, _ = torch.max(outputs.logits.softmax(axis=-1), dim=-1)
    predictions = outputs.logits.argmax(-1)
    labels = [[model.config.id2label[pred.item()] for pred in prediction] for prediction in predictions]
    results = _process_outputs(
        encoding=encoding,
        tokenizer=processor.tokenizer,
        labels=labels,
        scores=scores,
        image_size=image.size,
    )
    return results
