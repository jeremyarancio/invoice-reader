import logging
from typing import Any, Iterable, Dict
from PIL.Image import Image

from transformers import Pipeline


LOGGER = logging.getLogger(__name__)


class Predictor:

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        images: Iterable[Image],
        *args: Any, 
        **kwargs: Any
    ) -> Iterable[Dict]:
        """Predict total amount

        Args:
            image (Image): PIL image in 'RPG' format 

        Returns:
            Dict: 
            ```
            [{'score': 0.9999730587005615, 'answer': '18', 'start': 32, 'end': 32}]
            ```
        """
        LOGGER.info("Start inference.")
        pipe  = Pipeline("document-question-answering", model="impira/layoutlm-invoices", *args, **kwargs)
        predictions = pipe(
            images,
            "What is the invoice number?"
        )
        LOGGER.info(f"End prediction. Output: {predictions}")
        return predictions

        
