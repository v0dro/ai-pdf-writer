import cv2
import numpy as np

# Load the image
img = cv2.imread("letter_of_guarantee.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("dots2.png", cv2.IMREAD_GRAYSCALE)

def combine_rectangles(loc, h, w):
    combined = []
    prev_x = 0
    x =0
    in_rect = False
    for pt in zip(*loc):
        if not in_rect:
            start_y = pt[0]
            start_x = pt[1]
            prev_x = start_x
            width = 8
            in_rect = True
            continue
        
        y = pt[0]
        x = pt[1]

        if y != start_y:
            combined.append((start_y, start_x, h, width+20))
            in_rect = False
            continue

        if x <= start_x + width:
            print("x:", x, "y:", y, "start_x:", start_x, "width:", width)
            width += (x - prev_x)
        else:
            combined.append((start_y, start_x, h, width))
            in_rect = False
        prev_x = x

    combined.append((start_y, start_x, h, width+20))

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
combined_rectangles = combine_rectangles(loc, h, w)
print(combined_rectangles)

for pt in combined_rectangles:  # Switch columns and rows
    #cv2.rectangle(img, (pt[1], pt[0]-12), (pt[1] + pt[3], pt[0] -12 + pt[2]), (0, 255, 0), 2)
    cv2.putText(img, "$Blank$" , (pt[1]+20, pt[0]+h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),  2, 
                cv2.LINE_AA)
h2, w2 = template2.shape[::-1]

print("next template")
res2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(res2 >= threshold)
combined_rectangles = combine_rectangles(loc2, h, w2)

for pt in combined_rectangles:  # Switch columns and rows
    #cv2.rectangle(img, (pt[1], pt[0]-14), (pt[1] + pt[3], pt[0] -14+ pt[2]), (0, 255, 0), 2)
    cv2.putText(img, "$Blank$" , (pt[1]+20, pt[0]+h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),  2, 
                cv2.LINE_AA)

cv2.imwrite("result1.png", img)
