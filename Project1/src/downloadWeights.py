from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")

preprocessor.save_pretrained("./preprocessor")
model.save_pretrained("./model")