from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, \
    LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
from PIL import Image
from pdf2image import convert_from_path
import cv2
import pprint

# model_name = "nielsr/layoutlmv3-finetuned-funsd"
model_name = "nielsr/layoutlmv3-funsd-v2"
feature_extractor = LayoutLMv3FeatureExtractor(
    apply_ocr=True, ocr_lang="eng",
    tesseract_config="--psm 11")
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
# This is used for extracting text from an image of the PDF. It uses an OCR underneath
# and does not have any connection to the model.
processor = LayoutLMv3Processor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
pdf_images = convert_from_path("letter_of_guarantee.pdf")

for image in pdf_images:
    # image.save("letter_of_guarantee.png", "PNG")
    # gray = cv2.cvtColor(cv2.imread("letter_of_guarantee.png"), cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("processed_output.png", thresh)
    new_image = image
    # new_image = Image.open("processed_output.png")
    # new_image = new_image.convert("RGB")
    encoding = processor(
        new_image,
        max_length=1024,
        #padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    print(encoding.keys())
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

