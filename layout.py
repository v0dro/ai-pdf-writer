from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification,  LayoutLMv3ImageProcessor
from PIL import Image
import pprint
import re

model_name = "nielsr/layoutlmv3-finetuned-funsd"
# model_name = "nielsr/layoutlmv3-funsd-v2"
# model_name = "microsoft/layoutlmv3-base"
tokenizer = LayoutLMv3TokenizerFast(
    vocab_file="vocab.json", merges_file="merges.txt"
)
# This is used for extracting text from an image of the PDF. It uses an OCR underneath
# and does not have any connection to the model.
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
image_processor =  LayoutLMv3ImageProcessor(
    apply_ocr=True,
    ocr_lang="eng",
    tesseract_config="--psm 6")

new_image = Image.open("result1.png").convert("RGB")
processed_image = image_processor.preprocess(new_image)

word_list = list()
box_list = list()

def unwanted_word(word):
    if re.search(r'[a-zA-Z]', word):
        return False
    
    return True

for word, box in zip(processed_image['words'][0], processed_image['boxes'][0]):
    if not unwanted_word(word):
        word_list.append(word.strip())
        box_list.append(box)

print(word_list)
print(box_list)

encoding = tokenizer(text=word_list, boxes=box_list, return_tensors="pt")
outputs = model(**encoding)

logits = outputs.logits
predictions = logits.argmax(-1).squeeze().tolist()

labels = [model.config.id2label[pred] for pred in predictions]
decodes = [tokenizer.decode(e) for e in encoding['input_ids'].flatten()]

d = {k: v for k, v in zip(decodes, labels)}
pprint.pprint(d, sort_dicts=False)

