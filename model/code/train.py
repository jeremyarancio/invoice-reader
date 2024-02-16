import argparse
import logging
import os
import shutil
import sys
from distutils.util import strtobool
from typing import Tuple, Mapping

import torch.cuda
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    LayoutLMForTokenClassification, 
    LayoutLMTokenizerFast,
    Trainer, 
    TrainingArguments, 
    set_seed
)

from preprocess import preprocess


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],  # Necessary the catch training logging during training job²
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def copy_scripts(path: str) -> None:
    """Copy 
    * inference script, 
    * requirement 
    * preprocessing script,
    to model directory in the model artifact for later deployment and tracking.

    Args:
        path (str): SM_MODEL_DIR / code
    """
    def copyfile(script_name: str) -> None:
        shutil.copyfile(
        os.path.join(os.path.dirname(__file__), script_name),
        os.path.join(path, script_name)
        )

    os.makedirs(path, exist_ok=True)
    copyfile("inference.py")
    copyfile("requirements.txt")
    copyfile("preprocess.py")


def parse_args():
    parser = argparse.ArgumentParser()

    # Sagemaker environment
    parser.add_argument("--dataset_dir", type=str, default=os.getenv("SM_CHANNEL_TRAINING")) # "SM_CHANNEL_{name_data}"
    parser.add_argument("--output_dir", type=str, default=os.getenv("SM_MODEL_DIR"))

    #Training
    parser.add_argument("--pretrained_model_name", type=str, default="bigscience/bloom-3b", help="Name of the pretrained model to fine-tune.")
    parser.add_argument("--epochs", type=float, default=1, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--fp16", type=strtobool, default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--labels", type=list, help="List of token labels")
    
    args = parser.parse_known_args()
    return args


def train(args):
    """"""
    LOGGER.info("Start training script.")
    LOGGER.info(f"Parameters implemented:\n {args}")

    set_seed(args.seed)

    id2label = {v: k for v, k in enumerate(args.labels)}
    label2id = {k: v for v, k in enumerate(args.labels)}

    LOGGER.info(f"Load model and tokenizer from {args.pretrained_model_name}, with those labels {id2label} ")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(args.pretrained_model_name)
    model = LayoutLMForTokenClassification.from_pretrained(
        args.pretrained_model_name,
        num_labels=len(args.labels), 
        label2id=label2id, 
        id2label=id2label
    )

    LOGGER.info("Start preprocessing.")
    preprocessed_dataset = preprocess(
        dataset=load_from_disk(dataset_path=args.dataset_dir),
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
        learning_rate=args.lr,
        # logging & evaluation strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["test"]
    )

    LOGGER.info("Start training:")
    trainer.train()
    
    # Save
    LOGGER.info("Start saving tokenizer and model into model artifact.")
    tokenizer.save_pretrained(args.output_dir)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)

    LOGGER.info("Training script finished.")



if __name__ == "__main__":

    args, _ = parse_args()
    train(args)
    copy_scripts(os.path.join(args.output_dir, "code"))