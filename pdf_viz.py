from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast
from PIL import Image
from pdf2image import convert_from_path

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True, ocr_lang="eng")
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)
pdf_images = convert_from_path("letter_of_guarantee.pdf")

for image in pdf_images:
    image.save("le.png")
    encoding = processor(
        image,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    print(encoding.keys())

