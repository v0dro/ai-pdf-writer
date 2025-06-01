import cv2
import numpy as np

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("dots2.png", cv2.IMREAD_GRAYSCALE)

def combine_rectangles(loc, w, h):
    combined = []
    in_rect = False
    end_x = start_x = -1
    y_coord = -1
    for pt in zip(*loc[::-1]):
        if pt[1] != y_coord:
            in_rect = False
            if y_coord != -1 and end_x != -1:
                width = end_x - start_x
                combined.append((start_x, y_coord, width, h))

        if in_rect:
            end_x = pt[0]
        else:
            y_coord = pt[1]
            start_x = pt[0]
            in_rect = True

    # Return a list with tuples (x, y, w, h) for each combined rectangle.
    return combined

w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):  # Switch columns and rows
#     print("x:", pt[0], "y:", pt[1], "w:", w, "h:", h)
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

combined_rectangles = combine_rectangles(loc, w, h)
print(combined_rectangles)

for pt in combined_rectangles:  # Switch columns and rows
    print("x:", pt[0], "y:", pt[1], "w:", w, "h:", h)
    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0, 255, 0), 2)

w2, h2 = template2.shape[::-1]

#print("next template")
res2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(res2 >= threshold)
for pt in zip(*loc2[::-1]):  # Switch columns and rows
    #print("x:", pt[0], "y:", pt[1], "w:", w2, "h:", h2)
    cv2.rectangle(img, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 2)

cv2.imwrite("result.png", img)
