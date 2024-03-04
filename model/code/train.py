import argparse
import logging
import os
import shutil
import sys
from typing import Mapping
import json

from datasets import load_from_disk
from transformers import (
    LayoutLMForTokenClassification, 
    LayoutLMTokenizerFast,
    Trainer, 
    TrainingArguments, 
    set_seed
)
import numpy as np
import evaluate
import comet_ml

from preprocess import preprocess_dataset


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],  # Necessary the catch training logging during training job²
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

SM_TRAINING_ENV = json.loads(os.getenv("SM_TRAINING_ENV"))  # Need to be deserialized
SM_JOB_NAME = SM_TRAINING_ENV["job_name"]


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
    parser.add_argument("--pretrained_model_name", type=str, default="microsoft/layoutlm-base-uncased", help="Name of the pretrained model to fine-tune.")
    parser.add_argument("--epochs", type=float, default=1, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Eval batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    # parser.add_argument("--fp16", type=strtobool, default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--labels", nargs='+', type=str, help="List of labels.")
    parser.add_argument("--output_path", type=str, help="Training job artifact URI")
    
    args = parser.parse_known_args()
    return args


if __name__ == "__main__":

    args, _ = parse_args()

    LOGGER.info("Start training script.")
    LOGGER.info(f"Parameters implemented:\n {args}")

    set_seed(args.seed)

    id2label = {v: k for v, k in enumerate(args.labels)}
    label2id = {k: v for v, k in enumerate(args.labels)}
    
    LOGGER.info(f"Load model and tokenizer from {args.pretrained_model_name}, with labels {id2label} ")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(args.pretrained_model_name)
    model = LayoutLMForTokenClassification.from_pretrained(
        args.pretrained_model_name,
        num_labels=len(args.labels), 
        label2id=label2id, 
        id2label=id2label
    )

    LOGGER.info("Start preprocessing.")
    preprocessed_dataset = preprocess_dataset(
        dataset=load_from_disk(dataset_path=args.dataset_dir),
        tokenizer=tokenizer,
        labels_ref=args.labels
    )

    LOGGER.info(f"Preprocessed dataset: {preprocessed_dataset}")

    def compute_metrics(p) -> Mapping:
        """Trainer function to compute evaluation metrics."""

        experiment = comet_ml.get_global_experiment()
        LOGGER.info(f"Experiment name in compute_metrics: {experiment.get_name()}")
        metric = evaluate.load("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Keep only labeled tokens
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Transform into str for Seqeval
        true_predictions_str = [[args.labels[p] for p in prediction] for prediction in true_predictions]
        true_labels_str = [[args.labels[l] for l in label] for label in true_labels]
        
        # Seqeval requires BIO scheme for evaluation
        BIO_predictions = [["B-" + label if label != "O" else label for label in labels] for labels in true_predictions_str]
        BIO_labels = [["B-" + label if label != "O" else label for label in labels] for labels in true_labels_str]

        results = metric.compute(predictions=BIO_predictions, references=BIO_labels)

        LOGGER.info(f"Labels for confusion matrix: {[label for batch in true_labels for label in batch]}")
        LOGGER.info(f"Preds for confusion matrix: {[pred for batch in true_predictions for pred in batch]}")

        # Experiment tracking
        if experiment:
            epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
            experiment.set_epoch(epoch)
            experiment.log_confusion_matrix(
                y_true=[label for batch in true_labels for label in batch], # flatten [[0,0,1,0], [1,0,1,0]] -> [0,0,1,0,1,0,1,0]
                y_predicted=[pred for batch in true_predictions for pred in batch],
                file_name=f"confusion-matrix-epoch-{epoch}.json",
                labels=args.labels
            )

            LOGGER.info(f"Metric dict: {results}")
            # Extract each metric from results
            for metric_name, value in results.items():
                if isinstance(value, dict):
                    experiment.log_metrics(
                        dic=value,
                        prefix=metric_name,
                        epoch=epoch
                    )
                else:
                    experiment.log_metric(
                        name=metric_name, 
                        value=value,
                        epoch=epoch
                    )
        else:
            LOGGER.warning("Experiment from compute_metrics() is not found.")
        return results
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # fp16=args.fp16,
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
        eval_dataset=preprocessed_dataset["test"],
        compute_metrics=compute_metrics
    )

    LOGGER.info("Start training:")
    trainer.train()
    
    # Save
    LOGGER.info("Start saving tokenizer and model into model artifact.")
    tokenizer.save_pretrained(args.output_dir)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)

    # Log remote model with CometML and S3
    experiment = comet_ml.get_global_experiment()
    LOGGER.info(f"Experiment name after Transformers trainer: {experiment.get_name()}")
    experiment = comet_ml.ExistingExperiment(experiment_key=experiment.get_key())
    experiment.add_tags(["LayoutLM", "Test"])
    experiment.log_parameters(vars(args))

    model_uri = os.path.join(args.output_path, SM_JOB_NAME, "output/model.tar.gz")
    LOGGER.info(f"Training job uri: {model_uri}")
    experiment.log_remote_model(
        "LayoutLM", 
        model_uri, 
        sync_mode=False
    )

    LOGGER.info("Training script finished.")

    copy_scripts(os.path.join(args.output_dir, "code"))
