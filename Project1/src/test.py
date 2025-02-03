import torch
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification


preprocessor = EfficientNetImageProcessor.from_pretrained("./preprocessor")
model = EfficientNetForImageClassification.from_pretrained("./model")

image_path = "Project1/papaj.png"
output_path = "Project1/output.txt"

init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((224, 224))
inputs = preprocessor(init_image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
with open(output_path, "w") as file:
    file.write(model.config.id2label[predicted_label])