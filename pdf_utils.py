import requests
import cv2
import numpy as np
from pdf2image import convert_from_path

def download_letter_of_guarantee(f_name):
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    pr_form_url = "https://www.moj.go.jp/isa/content/930002537.pdf"
    r = requests.get(pr_form_url, headers=custom_headers)

    if r.status_code == 200:
        with open(f_name, "wb") as f:
            f.write(r.content)
    else:
        print(f"Failed to download the letter of guarantee. Status code: {r.status_code}")


def convert_pdf_to_png(pdf_path, png_path):
    images = convert_from_path(pdf_path)
    images[0].save(f"{png_path}", "PNG")

    print(f"Converted {pdf_path} to PNG format and saved as {png_path}.")

def _combine_rectangles(loc, h, w):
    combined = []
    prev_x = 0
    x = 0
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
            width += (x - prev_x)
        else:
            combined.append((start_y, start_x, h, width))
            in_rect = False
        prev_x = x

    combined.append((start_y, start_x, h, width+20))

    # Return a list with tuples (x, y, w, h) for each combined rectangle.
    return combined

def _find_rectangles_for_blanks(img, template, threshold, x_adjust, y_adjust):
    h, w = template.shape[::-1]

    # The co-ordinate system of openCV says that the origin (0, 0) is at 
    # the top-left corner of the image. The x-coordinates increase to the
    # right, and the y-coordinates increase downwards.
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # Therefore, the locations here are a 2D array where the rows indicate the
    # y-coordinates and the columns indicate the x-coordinates.
    loc = np.where(res >= threshold)
    combined_rectangles = _combine_rectangles(loc, h, w)

    # Adjust the size of the rectangle.
    combined_rectangles = list(map(lambda x: (x[0] + x_adjust, x[1] + y_adjust, x[2], x[3]), combined_rectangles))

    return combined_rectangles

def find_form_blanks(png_path, write_image=False):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)
    template2 = cv2.imread("dots2.png", cv2.IMREAD_GRAYSCALE)
    threshold = 0.9

    combined_rectangles = _find_rectangles_for_blanks(img, template, threshold, 20, 20)
    combined_rectangles += _find_rectangles_for_blanks(img, template2, threshold, 15, 15)

    if write_image:
        i = 0
        for pt in combined_rectangles:  # Switch columns and rows
            # cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + pt[3], pt[0] + pt[2]), (0, 255, 0), 2)
            cv2.putText(img, f"{i}" , (pt[1], pt[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),  2, 
                        cv2.LINE_AA)
            i+= 1
            
        cv2.imwrite(f"overlay_{png_path}", img)

    return combined_rectangles