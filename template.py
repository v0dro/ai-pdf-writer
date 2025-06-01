import cv2
import numpy as np

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("dots2.png", cv2.IMREAD_GRAYSCALE)

def combine_rectangles(loc, h, w):
    combined = []
    in_rect = False
    width = 0
    y = None
    x = None
    for pt in zip(*loc):
        # print("x:", x, " y: ", y, " width: ", width, " w: ", w)
        if not in_rect:
            start_y = pt[0]
            start_x = pt[1]
            width = 0
            in_rect = True
        else:
            prev_y = y
            prev_x = x
            y = pt[0]
            x = pt[1]


            if x <= start_x + width:
                print("start_y:", start_y, "start_x:", start_x, "y:", y, "x:", x, "width:", width, "w:", w, "start_x + width:", start_x + width, " x-start_x:", x-start_x)

                width += (x - start_x)
            else:
                in_rect = False
                combined.append((start_y, start_x, h, width + w))

    # Return a list with tuples (x, y, w, h) for each combined rectangle.
    return combined

h, w = template.shape[::-1]

# The co-ordinate system of openCV says that the origin (0, 0) is at the top-left corner of the image.
# The x-coordinates increase to the right, and the y-coordinates increase downwards.
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9

# Therefore, the locations here are a 2D array where the rows indicate the
# y-coordinates and the columns indicate the x-coordinates.
loc = np.where(res >= threshold)
# for pt in zip(*loc):  # Switch columns and rows
#     print("x:", pt[0], "y:", pt[1], "w:", w, "h:", h)
#     cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + h, pt[0] + w), (0, 255, 0), 2)

combined_rectangles = combine_rectangles(loc, h, w)
print(combined_rectangles)

for pt in combined_rectangles:  # Switch columns and rows
    # print("x:", pt[0], "y:", pt[1], "w:", w, "h:", h)
    cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + pt[3], pt[0] + pt[2]), (0, 255, 0), 2)

# w2, h2 = template2.shape[::-1]

# #print("next template")
# res2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
# loc2 = np.where(res2 >= threshold)
# for pt in zip(*loc2[::-1]):  # Switch columns and rows
#     #print("x:", pt[0], "y:", pt[1], "w:", w2, "h:", h2)
#     cv2.rectangle(img, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 2)

cv2.imwrite("result.png", img)
