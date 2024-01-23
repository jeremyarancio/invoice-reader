import logging
from typing import Dict, Mapping
import json
import re

from datasets import load_dataset

from estimator import Estimator


LOGGER = logging.getLogger(__name__)


class Validator:
    """Validation class"""

    def __init__(self, validation_dataset: str) -> None:
        self.validation_dataset = validation_dataset

    def __call__(
            self,
            estimator: Estimator,
            streaming: bool = True
        ) -> Dict:
        """"""
        validation = []
        test_dataset = load_dataset(
            path=self.validation_dataset,
            split="test",
            streaming=streaming
        )
        LOGGER.info(f"Dataset for validation: {test_dataset}")
        updated_dataset = test_dataset.map(
            self.map_function,
            remove_columns="ground_truth"
        )
        for element in updated_dataset:
            prediction = estimator(element["image"])[0]
            LOGGER.info(f"Prediction during inference: {prediction}")
            total_gross_worth = prediction["answer"]
            total_gross_worth = self.process_amount(total_gross_worth)
            LOGGER.debug(f"gt_total_gross_worth: {element['gt_total_gross_worth']}; predicted_total_gross_worth: {total_gross_worth}")
            validation.append(total_gross_worth == element['gt_total_gross_worth'])
        return validation
    
    def map_function(self, element: Mapping) -> Mapping:
        """"""
        ground_truth = json.loads(element["ground_truth"])
        gt_total_gross_worth = ground_truth["gt_parse"]["summary"]["total_gross_worth"]
        gt_total_gross_worth = self.process_amount(gt_total_gross_worth)
        return {"gt_total_gross_worth": gt_total_gross_worth}

    @staticmethod
    def process_amount(text: str) -> str:
        """Transform amounts (such as gross amount, net amout, ...) into an uniform for comparison.
        
        Example: '$45,54' or '$ 45. 54' -> 45.54
        """
        pattern = re.compile("(\d+)[.,](\d+)")
        matches = pattern.search(text).groups()
        text = matches[0] + "." + matches[1]
        return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    validator = Validator(validation_dataset="katanaml-org/invoices-donut-data-v1")
    estimator = Estimator()
    validation = validator(estimator=estimator)