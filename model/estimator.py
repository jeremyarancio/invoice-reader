import logging
from typing import Any, Iterable, Dict
from PIL.Image import Image

from transformers import pipeline


LOGGER = logging.getLogger(__name__)


class Estimator:

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        image: Image,
        *args: Any, 
        **kwargs: Any
    ) -> Iterable[Dict]:
        """Predict "total gross worth"

        Args:
            image (Image): PIL image in 'RPG' format 

        Returns:
            Dict: 
            ```
            [{'score': 0.9999730587005615, 'answer': '18', 'start': 32, 'end': 32}]
            ```
        """
        LOGGER.info("Start inference.")
        pipe  = pipeline("document-question-answering", model="impira/layoutlm-invoices", *args, **kwargs)
        predictions = pipe(
            image,
            "What is the total gross worth of this invoice?"
        )
        LOGGER.info(f"End prediction. Output: {predictions}")
        return predictions

        
