from pdf_utils import download_letter_of_guarantee, \
    convert_pdf_to_png, find_form_blanks
from ai_chat import letter_of_guarantee_chat

if __name__ == "__main__":
    pdf_file = "letter_of_guarantee.pdf"
    png_file = "letter_of_guarantee.png"
    download_letter_of_guarantee(pdf_file)
    convert_pdf_to_png(pdf_file, png_file)
    find_form_blanks(png_file)
    user_form_fields = letter_of_guarantee_chat()
