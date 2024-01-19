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