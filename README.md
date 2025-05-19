# ai-pdf-writer

Install tesseract on linux using the following command:
```
sudo apt -y install tesseract-ocr tesseract-ocr-jpn libtesseract-dev libleptonica-dev tesseract-ocr-script-jpan tesseract-ocr-script-jpan-vert 
```

Then, head over to https://github.com/tesseract-ocr/tessdata/tags and download the latest version of tessdata and put it in the `$HOME/.tessdata`. Set `TESSDATA_PREFIX` to `$HOME/.tessdata` in your `.bashrc` file.