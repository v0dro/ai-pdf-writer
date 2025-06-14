import requests
from pdf2image import convert_from_path

def download_letter_of_guarantee(f_name):
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    pr_form_url = "https://www.moj.go.jp/isa/content/930002537.pdf"
    r = requests.get(pr_form_url, headers=custom_headers)

    if r.status_code != 200:
        with open(f_name, "wb") as f:
            f.write(r.content)
    else:
        print(f"Failed to download the letter of guarantee. Status code: {r.status_code}")


def convert_pdf_to_png(pdf_path, png_path):
    images = convert_from_path(pdf_path)
    images[0].save(f"{png_path}", "PNG")

    print(f"Converted {pdf_path} to PNG format and saved as {png_path}.")