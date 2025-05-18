import requests

def get_pr_form(f_name):
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    pr_form_url = "https://www.moj.go.jp/isa/content/930002835.pdf"
    r = requests.get(pr_form_url, headers=custom_headers)
    if r.status_code != 200:
        with open(f_name, "wb") as f:
            f.write(r.content)

get_pr_form("pr_form.pdf")