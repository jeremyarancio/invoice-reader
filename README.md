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

## If I had to start over

* Objectives
* Data
* Models design
* Prepare Benchmark (should not be relative to training data)
* Evaluate existing solutions
* ML
  * Prepare sample of data for first model
  * Preprocess
  * Cloud:
    * Explore model training using Cloud
    * Inference
  * Prepare Estimator + Inference + Preprocess + W&B (or similar)
  * Sagemaker Training jobs
  * Model registry
  * Deployment 


## Deployment

* If `inference.py` is added as an `entry_point`, it will repack the S3 artifact which takes a lot of time
* Maybe it's better to have the script ready during the training stage. 
* Error message: 

```
sagemaker.exceptions.UnexpectedStatusException: Error hosting endpoint huggingface-pytorch-inference-2024-03-07-10-01-23-289: Failed. Reason: Failed to extract model data archive from URL "s3://sagemaker-eu-central-1-265890761777/huggingface-pytorch-inference-2024-03-07-09-41-59-363/model.tar.gz". The model data archive is too large. Please reduce the size of the model data archive or move to an instance type with more memory.. Try changing the instance type or reference the troubleshooting page https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-troubleshooting.html
```
* It looks like the decompress -> add file -> compress is done on my computer, that's why it's so slow (25min / 2.5GB)
* Possibility to repack model using Lambda or Lambda step?


# AWS Deep Learning Containers

https://github.com/aws/deep-learning-containers/blob/master/available_images.md