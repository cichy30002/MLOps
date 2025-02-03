import boto3
import torch
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

preprocessor = EfficientNetImageProcessor.from_pretrained("./preprocessor")
model = EfficientNetForImageClassification.from_pretrained("./model")

s3 = boto3.client('s3')
image_path = "/tmp/input_image.png"
output_path = "/tmp/output.txt"

def lambda_handler(event, context):

    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    object_key = event["Records"][0]["s3"]["object"]["key"]
    s3.download_file(bucket_name, object_key, image_path)

    init_image = Image.open(image_path).convert("RGB")

    init_image = init_image.resize((224, 224))
    inputs = preprocessor(init_image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    with open(output_path, "w") as file:
        file.write(model.config.id2label[predicted_label])

    with open(output_path, 'rb') as data:
        s3.put_object(Bucket=bucket_name, Key='output.txt', Body=data)

    return {
        'statusCode': 200,
        'body': 'I saved the result to output.txt'
    }
