import cv2
from PIL import Image, ImageDraw, ImageFont

img = cv2.imread('letter_of_guarantee.png', cv2.IMREAD_GRAYSCALE)

# Threshold to binary
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Find contours (line positions)
contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image = Image.open('letter_of_guarantee.png').convert("RGB")
draw = ImageDraw.Draw(image)
# font = ImageFont.truetype("arial.ttf", 20)

for cnt in contours:
    print(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    draw.text((x, y - 5), "<BLANK>", fill=(0, 0, 0))  # You could use '•' or '<BLANK>' too
    
image.save("processed_form.png")