# invoice-reader
An app to read and store invoices using Machine Learning and AWS


# Installation notes

LayoutLM requires tesseract and pytesseract.

It seemed there was an issue with conda and libtiff.so.5. What worked:

```bash
conda install -c conda-forge tesseract 
pip install pytesseract
```

Basic installation from [HF article](https://www.philschmid.de/fine-tuning-layoutlm#4-run-inference-and-parse-form):

```bash
sudo apt install -y tesseract-ocr
pip install pytesseract transformers datasets seqeval tensorboard
```

# Label studio

## Type of data to import

[Label studio import data OCR](https://labelstud.io/guide/predictions.html#Import-OCR-pre-annotations)

```json
[{
  "data": {
    "image": "/static/samples/sample.jpg" 
  },

  "predictions": [{
    "model_version": "one",
    "score": 0.5,
    "result": [
      {
        "id": "result1",
        "type": "rectanglelabels",        
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,          
          "x": 4.98, "y": 12.82,
          "width": 32.52, "height": 44.91,
          "rectanglelabels": ["Airplane"]
        }
      },
      {
        "id": "result2",
        "type": "rectanglelabels",        
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,          
          "x": 75.47, "y": 82.33,
          "width": 5.74, "height": 7.40,
          "rectanglelabels": ["Car"]
        }
      },
      {
        "id": "result3",
        "type": "choices",
        "from_name": "choice", "to_name": "image",
        "value": {
          "choices": ["Airbus"]
      }
    }]
  }]
}]
```

## COnvert pdf into image

```bash
apt-get install poppler-utils
pip install pdf2image
```

## Good AWS Sagemaker / MLOps resources

* [Create SageMaker Pipelines for training, consuming and monitoring your batch use cases](https://aws.amazon.com/blogs/machine-learning/create-sagemaker-pipelines-for-training-consuming-and-monitoring-your-batch-use-cases/)
* [Automate Machine Learning Workflows](https://aws.amazon.com/tutorials/machine-learning-tutorial-mlops-automate-ml-workflows/)
* https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipeline-compare-model-versions/notebook.ipynb
* 