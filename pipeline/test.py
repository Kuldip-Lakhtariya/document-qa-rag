from extract_text import extract_text_from_pdf

extracted_pages = extract_text_from_pdf(r"C:\Users\Admin\Downloads\document_pdf.pdf")
print(len(extracted_pages))
print(extracted_pages[0]["text"][:300])

for page_data in extracted_pages:
    print(f"--- Page {page_data['page']} ---")
    print(page_data["text"][:500])
    print()