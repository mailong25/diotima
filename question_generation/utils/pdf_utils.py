import fitz

def pdf_to_text(pdf_path, p_start, p_end):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(p_start - 1, p_end):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text
