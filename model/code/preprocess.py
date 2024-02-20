from typing import Mapping, List
import logging

from datasets import DatasetDict, disable_caching
from transformers import AutoTokenizer, BatchEncoding

disable_caching()


LOGGER = logging.getLogger(__name__)


def preprocess(
    dataset: DatasetDict, tokenizer: AutoTokenizer, labels_ref: List[str]
) -> DatasetDict:
    """
    Features: ['input_ids', 'token_type_ids', 'attention_mask', 'bbox', "labels"])
    """
    LOGGER.info(f"Dataset info: {dataset}")
    processed_dataset = dataset.map(
        batched_map_fn,
        fn_kwargs={"tokenizer": tokenizer, "labels_ref": labels_ref},
        remove_columns=dataset["train"].column_names,
        keep_in_memory=False,
        batched=True
    ).with_format("torch")

    # assert encoding["input_ids"].shape[-1] == tokenizer.model_max_length
    # assert len(encoding["labels"]) == tokenizer.model_max_length
    # assert len(encoding["bbox"]) == tokenizer.model_max_length

    return processed_dataset


def batched_map_fn(
    element: Mapping, 
    tokenizer: AutoTokenizer, 
    labels_ref: List[str]
) -> Mapping:
    """https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__"""
    encoding: BatchEncoding = tokenizer(
        element["words"],
        truncation=True,
        return_tensors="pt",
        padding="max_length",      # Max_length of the model (here 512)
        is_split_into_words=True,  # Token classification
    )
    encoding["labels"] = align_labels(
        encoding=encoding, 
        batched_labels=element["labels"], 
        labels_ref=labels_ref
    )
    encoding["bbox"] = align_bboxes(
        encoding=encoding,
        batched_bboxes=element["bboxes"],
        widths=element["original_width"],
        heights=element["original_height"],
        max_tokens=tokenizer.model_max_length,
    )
    return encoding


def align_labels(
    encoding, 
    batched_labels: List[List[str]], 
    labels_ref: List[str]
):
    """https://huggingface.co/docs/transformers/v4.37.2/en/tasks/token_classification#preprocess
    """
    label2id = {k: v for v, k in enumerate(labels_ref)}

    batched_aligned_labels = []
    for i, labels in enumerate(batched_labels):
        label_ids = []
        word_ids = encoding.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[labels[word_idx]])  # Transform label into it's respective ID
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        batched_aligned_labels.append(label_ids)
    return batched_aligned_labels


def align_bboxes(
    encoding,
    batched_bboxes,
    widths,
    heights,
    max_tokens,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
) -> List[List[List[int]]]:
    """"""
    batched_aligned_bboxes = []
    for i, bboxes in enumerate(batched_bboxes):
        aligned_bboxes = []
        word_ids = encoding.word_ids(batch_index=i)  # Map tokens to their respective word.
        word_ids = [word_idx for word_idx in word_ids if word_idx is not None]  # Remove None corresponding to CLS and EOS
        for word_idx in word_ids:
            normalized_bbox = normalize_bbox(
                bbox=bboxes[word_idx], width=widths[i], height=heights[i]
            )
            aligned_bboxes.append(normalized_bbox)
        aligned_bboxes = [cls_token_box] + aligned_bboxes + [sep_token_box] # Add bboxes for CLS and EOS
        aligned_bboxes += [pad_token_box] * (max_tokens - len(aligned_bboxes)) # Padding
        aligned_bboxes = aligned_bboxes[:max_tokens]
        batched_aligned_bboxes.append(aligned_bboxes)
    return batched_aligned_bboxes


def normalize_bbox(bbox, width: int, height: int) -> List[int]:
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
