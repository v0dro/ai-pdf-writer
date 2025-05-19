from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, LayoutLMv3FastTokenizer
from transformers.model.fnet.modeling_fnet import apply_chunking_to_forward
from PIL import Image
from pdf2image import convert_from_path

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True, ocr_lang="en")
tokenizer = LayoutLMv3FastTokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)
pdf_images = convert_from_path("letter_of_guarantee.pdf")

