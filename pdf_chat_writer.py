"""
pdf_chat_writer.py

This script automates the process of filling out a letter of guarantee PDF form using conversational AI.
It performs the following steps:
1. Downloads the letter of guarantee PDF.
2. Converts the PDF to a PNG image.
3. Detects blank fields in the form image.
4. Interacts with the user via a chatbot to collect required form data.
5. Overlays the collected data onto the detected blank fields in the form image.
6. Saves the resulting image with the overlaid data.

Dependencies:
- pdf_utils: For PDF download, conversion, and blank detection.
- ai_chat: For conversational data collection.
"""

from pdf_utils import download_letter_of_guarantee, \
    convert_pdf_to_png, find_form_blanks
from ai_chat import letter_of_guarantee_chat
import cv2

def fit_text_to_rectangle(image, text, rect_x, rect_y, rect_width, rect_height, 
                         font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    """
    Scale text to fit within a given rectangle and draw it on the image.
    
    Args:
        image: Input image (numpy array)
        text: Text string to draw
        rect_x, rect_y: Top-left corner of rectangle
        rect_width, rect_height: Dimensions of rectangle
        font: OpenCV font type
        thickness: Text thickness
        
    Returns:
        Modified image with text drawn
        Final font scale used
    """
    
    # Start with a reasonable font scale
    font_scale = 1.0
    
    # Get text size with current scale
    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

    while rect_width < text_w:
        font_scale -= 0.01
        (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.putText(image, text, (rect_x, rect_y), 
                font, font_scale, (0,0,0), 2)
    
    return image, font_scale

if __name__ == "__main__":
    pdf_file = "letter_of_guarantee.pdf"
    png_file = "letter_of_guarantee.png"
    download_letter_of_guarantee(pdf_file)
    convert_pdf_to_png(pdf_file, png_file)
    form_rectangles = find_form_blanks(png_file, False)
    user_form_fields = letter_of_guarantee_chat()

    form_fields_order = [
        "full_name", # 0
        "guarantor.address_in_japan", # 1
        "guarantor.guarantor_phone_number", # 2
        "guarantor.place_of_employment", # 3
        "guarantor.occupation_phone_number", # 4
        "guarantor.nationality", # 5
        "guarantor.status_of_residence", # 6
        "guarantor.guarantor_relationship", # 7
        "date", # 8
        "nationality", # 9
        "guarantor.name", # 10
        "guarantor.signature" # 11
    ]

    img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
    for rectangle_index, field in enumerate(form_fields_order):
        if "signature" in field:
            continue
        keys = field.split(".")
        rectangle = form_fields_order[rectangle_index]
        user_data = user_form_fields

        for k in keys:
            user_data = user_data[k]

        pt = form_rectangles[rectangle_index]

        img, _ = fit_text_to_rectangle(img, user_data, pt[1], pt[0], pt[3], pt[2])

    cv2.imwrite(f"overlay.png", img)