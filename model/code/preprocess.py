from typing import Mapping, List
import logging

from datasets import load_from_disk, DatasetDict, disable_caching
from transformers import AutoTokenizer, BatchEncoding
disable_caching()


LOGGER = logging.getLogger(__name__)


ID2LABEL = {v: k for v, k in enumerate(LABELS)}
LABEL2ID = {k: v for v, k in enumerate(LABELS)}


def preprocess(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Features: ['input_ids', 'token_type_ids', 'attention_mask', 'bbox'])
    """
    LOGGER.info("Start preprocessing step.")
    dataset: DatasetDict = load_from_disk(dataset_path, keep_in_memory=False)
    LOGGER.info(f"Dataset info: {dataset}")
    processed_dataset = dataset.map(
        map_fn, 
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
        keep_in_memory=False
    )
    return processed_dataset


def map_fn(element: Mapping, tokenizer: AutoTokenizer) -> Mapping:
    """https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__"""
    encoding: BatchEncoding = tokenizer(
        element["words"], 
        truncation=True, 
        return_tensors="pt", 
        is_split_into_words=True # Token classification
    )
    encoding["labels"] = align_labels(
        encoding=encoding, 
        labels=element["labels"]
    )
    encoding["bbox"] = align_bboxes(
        encoding=encoding,
        bboxes=element["bboxes"],
        width=element["original_width"],
        height=element["original_height"]
    )
    return encoding    


def align_labels(encoding, labels: List):
    """"""
    word_ids = encoding.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(LABEL2ID[labels[word_idx]]) # Transform label into it's respective ID
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


def align_bboxes(encoding, bboxes, width, height):
    """"""
    aligned_bboxes = []
    word_ids = encoding.word_ids()  # Map tokens to their respective word.
    word_ids = [word_idx for word_idx in word_ids if word_idx is not None] # Remove None corresponding to CLS and EOS
    for word_idx in word_ids:
        normalized_bbox = normalize_bbox(
            bbox=bboxes[word_idx], 
            width=width,
            height=height
        )
        aligned_bboxes.append(normalized_bbox)
    aligned_bboxes = [[0, 0, 0, 0]] + aligned_bboxes + [[1000, 1000, 1000, 1000]] # Add bboxes for CLS and EOS
    return aligned_bboxes


def normalize_bbox(bbox, width: int, height: int) -> List[int]:
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


if __name__ == "__main__":
    path = "s3://invoice-reader-project/data/training/datasets/dataset_ocr_v1/"
    preprocess(path)