import logging
from typing import Mapping, Iterable
import json
import re

from datasets import (
    load_dataset,
    concatenate_datasets,
    disable_caching,
    IterableDataset,
)

from estimator import Estimator


LOGGER = logging.getLogger(__name__)
disable_caching()


def calculate_accuracy(preds: Iterable, tests: Iterable) -> float:
    """Evaluation accuracy.

    Args:
        preds (Iterable): Predictions
        tests (Iterable): Ground truth

    Returns:
        float: accuracy
    """
    validation = [1 if pred == test else 0 for pred, test in zip(preds, tests)]
    acc = sum(validation) / len(validation)
    return acc


def process_amount(text: str) -> str:
    """Transform amounts (such as gross amount, net amout, ...) into an uniform for comparison.

    Example: '$45,54' or '$ 45. 54' -> 45.54
    """
    pattern = re.compile("(\d+)[.,](\d+)")
    matches = pattern.search(text).groups()
    text = matches[0] + "." + matches[1]
    return text


def load_validation_dataset(path: str, streaming: bool) -> IterableDataset:
    """Load Test and Validation datasets from HF hub and concatenate them.

    Args:
        path (str): Dataset hub path
        streaming (bool): Transform Dataset into IterableDataset (https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.IterableDataset)

    Returns:
        IterableDataset
    """
    validation_ds = load_dataset(path, split="validation", streaming=streaming)
    test_ds = load_dataset(path, split="test", streaming=streaming)
    return concatenate_datasets([validation_ds, test_ds])


def map_function(element: Mapping) -> Mapping:
    """Specific to "katanaml-org/invoices-donut-data-v1"

    Args:
        element (Mapping): Element of the Dataset

    Returns:
        Mapping: Transformed element
    """
    ground_truth = json.loads(element["ground_truth"])
    # TODO Issue within the dataset. Consider modifying manually the data.
    if "summary" in ground_truth["gt_parse"].keys():
        gt_total_gross_worth = ground_truth["gt_parse"]["summary"]["total_gross_worth"]
    elif "total_gross_worth" in ground_truth["gt_parse"].keys():
        gt_total_gross_worth = ground_truth["gt_parse"]["total_gross_worth"]
    else:
        LOGGER.warning(
            f"There's no total_gross_worth found in the following element: {element}"
        )
    gt_total_gross_worth = process_amount(gt_total_gross_worth)
    return {"gt_total_gross_worth": gt_total_gross_worth}


def filter_function(element: Mapping) -> bool:
    """Specific to "katanaml-org/invoices-donut-data-v1". `total_gross_worth` can be missing.

    Args:
        element (Mapping): Element of the Dataset

    Returns:
        bool: If True, keep the element. Drop it otherwise.
    """
    ground_truth = json.loads(element["ground_truth"])
    if "total_gross_worth" in ground_truth["gt_parse"]:
        return True
    if "summary" in ground_truth["gt_parse"]:
        if "total_gross_worth" in ground_truth["gt_parse"]["summary"].keys():
            return True
    return False


class Validation:
    """Validation class. Run the estimator on the validation dataset."""

    def __init__(self, validation_dataset_path: str) -> None:
        self.validation_dataset_path = validation_dataset_path

    def evaluate(
        self, estimator: Estimator, streaming: bool = True
    ) -> Mapping[str, float]:
        """Evaluate the model on the evaluation dataset.

        Args:
            estimator (Estimator): Model performing the parsing.
            streaming (bool, optional): Load the evaluation dataset as an IterableDataset(https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.IterableDataset).
            Defaults to True.

        Returns:
            Mapping[str, float]: evaluation metrics
        """
        preds = []
        tests = []
        scores = []

        evaluation_dataset = load_validation_dataset(
            self.validation_dataset_path, streaming=streaming
        )

        LOGGER.info(f"Dataset for validation: {evaluation_dataset}")
        updated_dataset = evaluation_dataset.filter(filter_function).map(
            map_function, remove_columns="ground_truth"
        )
        for element in updated_dataset:
            prediction = estimator(element["image"])[0]
            total_gross_worth = process_amount(prediction["answer"])
            preds.append(total_gross_worth)
            tests.append(element["gt_total_gross_worth"])
            scores.append(prediction["score"])

        LOGGER.info(
            f"Comparison (prediction, ground_truth, score):\n {list(zip(preds, tests, scores))}"
        )
        acc = calculate_accuracy(preds=preds, tests=tests)
        LOGGER.info(f"Evaluation accuracy: {acc}")
        return {"acc": acc}


# if __name__ == "__main__":  
#     logging.basicConfig(level=logging.INFO)
#     validation = Validation(
#         validation_dataset_path="katanaml-org/invoices-donut-data-v1"
#     )
#     estimator = Estimator()
#     acc = validation.evaluate(estimator=estimator)
#     print(acc)
