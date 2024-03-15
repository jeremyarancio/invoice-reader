import sagemaker

image = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region="eu-central-1",
    image_scope="inference",
    version="1.13",
    instance_type="ml.c5.xlarge",
)
print(image)
