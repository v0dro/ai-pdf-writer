import cv2

img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)

threshold_value = 230
_, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imwrite("a.png", thresh)