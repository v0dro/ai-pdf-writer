from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, \
    LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
from PIL import Image, ImageOps 
from pdf2image import convert_from_path
import cv2
import pprint

model_name = "nielsr/layoutlmv3-finetuned-funsd"
# model_name = "nielsr/layoutlmv3-funsd-v2"
# model_name = "microsoft/layoutlmv3-base"
feature_extractor = LayoutLMv3FeatureExtractor(
    apply_ocr=True, ocr_lang="eng",
    tesseract_config="--psm 6")
tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name)
# This is used for extracting text from an image of the PDF. It uses an OCR underneath
# and does not have any connection to the model.
processor = LayoutLMv3Processor.from_pretrained(model_name)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

new_image = Image.open("result1.png").convert("RGB")
encoding = processor(
    new_image,
    max_length=1024,
    truncation=True,
    return_tensors="pt"
)
features = feature_extractor(new_image)

print(features['pixel_values'][0].shape)
print(features.keys())
words = features['words'][0]
boxes = features['boxes'][0]
for w, b in zip(words, boxes):
    enc = tokenizer(text=[w], boxes=[b]).input_ids
    print("word: ", w, " tokens: ", enc, " decode: ", tokenizer.decode(enc))

outputs = model(**encoding)
logits = outputs.logits
predictions = logits.argmax(-1).squeeze().tolist()

print(len(words))
labels = [model.config.id2label[pred] for pred in predictions]
print(len(labels))
print(words)
print(labels)
decodes = [tokenizer.decode(e) for e in encoding['input_ids'].flatten()]

d = {k: v for k, v in zip(decodes, labels)}
pprint.pprint(d, sort_dicts=False)

