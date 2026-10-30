import boto3

BUCKET_NAME = "plant-disease-dataset18"

s3 = boto3.client(
    "s3",
    region_name="ap-south-1"
)