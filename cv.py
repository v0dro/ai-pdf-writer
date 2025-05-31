import cv2

# #Create default parametrization LSD
# lsd = cv2.createLineSegmentDetector(0)

# #Detect lines in the image
# lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

# #Draw detected lines in the image
# drawn_img = lsd.drawSegments(img,lines)
# # cv2.imshow("LSD",drawn_img )
# # cv2.waitKey(0)

# threshold_value = 230
# _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
# cv2.imwrite("a.png", drawn_img)

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)

# # Invert and binarize
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite("thresh.png", thresh)

blur = cv2.GaussianBlur(thresh,(5,5), 0)
cv2.imwrite("thresh.png", thresh)

import pytesseract
from PIL import Image, ImageDraw

# Load your PNG image
image_path = 'thresh.png'
image = Image.open(image_path)
draw = ImageDraw.Draw(image)    

# Run OCR and get detailed data
data = pytesseract.image_to_data(
    image, 
    config='--psm 6',  # Assume a single uniform block of text
    output_type=pytesseract.Output.DICT)

# Extract bounding boxes
bounding_boxes = []
for i in range(len(data['text'])):
    word = data['text'][i]
    if word.strip() != "":  # Only include non-empty words
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        print(word, len(word), "w:", w, 'h:', h)
        bounding_boxes.append({'word': word, 'bbox': (x, y, x + w, y + h)})
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

image.show()
image.save("output_with_bboxes.png")