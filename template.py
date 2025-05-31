import cv2
import numpy as np

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("dots2.png", cv2.IMREAD_GRAYSCALE)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Switch columns and rows
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

w2, h2 = template2.shape[::-1]

res2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(res2 >= threshold)
for pt in zip(*loc2[::-1]):  # Switch columns and rows
    cv2.rectangle(img, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 2)

cv2.imwrite("result.png", img)
