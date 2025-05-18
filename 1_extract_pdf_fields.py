from pypdf import PdfReader

reader = PdfReader("pr_form.pdf")
fields = reader.get_fields()

for field_name, field_data in fields.items():
    print(f"{field_name}: {field_data.get('/V')}")