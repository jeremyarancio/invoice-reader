import logging
from typing import Dict
import json
from tqdm import tqdm

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
        validation = []
        test_dataset = load_dataset(
            path=self.validation_dataset,
            split="test",
            streaming=streaming
        )
        LOGGER.info(f"Dataset for validation: {test_dataset}")
        for row in tqdm(test_dataset, ):
            LOGGER.info(f"A row in the dataset: {row}")
            ground_truth = json.loads(row["ground_truth"])
            gt_total_gross_worth = ground_truth["gt_parse"]["summary"]["total_gross_worth"]
            gt_total_gross_worth = self._process_ground_truth(gt_total_gross_worth)
            predictions = estimator(row["image"])
            for prediction in predictions:
                total_gross_worth = prediction["answer"]
                validation.append(total_gross_worth == gt_total_gross_worth)
        return validation
    
    @staticmethod
    def _process_ground_truth(text: str) -> str:
        text = text.replace("$ ", "")
        return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    validator = Validator(validation_dataset="katanaml-org/invoices-donut-data-v1")
    estimator = Estimator()
    validator(estimator=estimator)
