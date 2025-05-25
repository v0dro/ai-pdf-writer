import cv2
import numpy as np

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)

# Invert and binarize
_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# Use a small horizontal kernel to enhance dots
# Create a small horizontal kernel to enhance dotted lines of size (5,1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
# Dilate the thresholded image to connect dots into solid lines
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
# Convert grayscale to color image for visualization
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
line_boxes = []

# Draw boxes around likely dotted lines

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    # if w > 100 and 1 < aspect_ratio < 100:  # Heuristic filter for dotted horizontal lines
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
    line_boxes.append((x, y, x + w, y + h))

print(line_boxes)
cv2.imwrite("img_color.png", img_color)
cv2.imwrite("dilated.png", dilated)

# Detect bounding box and remove it.
# Find any dotted lines and make them solid
# Save the co-ordinates of the soid lines
# Put <BLANK> in the lines so that they can be picked up an OCR

def remove_bounding_box():
    pass