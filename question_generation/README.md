#  Book Outline & RAG Extraction Tool

This repository provides tools for:

✅ **Generating a structured JSON outline of a book** by parsing the table of contents and extracting text.  
✅ **Extracting relevant text snippets** from the book JSON using a topic or subtopic.

---

### Setup
Make sure you put your API key in the config.py fie. This source code using openAI API.

##  1️⃣ Generate Book JSON

Use the **`build_book_json.py`** script to create a `book.json` file that contains structured data (chapters, subtopics, and associated text).

### 🔧 Usage

```bash
python build_book_json.py \
  --pdf_path /path/to/book.pdf \
  --tab_start 6 \
  --tab_end 9 \
  --page_offset 9 \
  --output_path resources/book.json
```

## 💡 Arguments
```
--pdf_path	Path to the PDF file.
--tab_start	Start page index (in the PDF) of the table of contents.
--tab_end	End page index (in the PDF) of the table of contents.
--page_offset	Offset to map the TOC page numbers to the actual PDF page indices (e.g., if TOC page 1 is PDF 13).
--output_path	Path to save the generated book.json file.
```

## 🔍 2️⃣ Extract Relevant Text (RAG)
Use the rag.py module to extract relevant content from book.json based on a topic or subtopic.

🔧 Example Usage
```
from rag import get_relevant_text
import json

book = json.load(open("resources/book.json"))

topic = "Sensory"
subtopic = "Taste"

relevant_text, selected_units = get_relevant_text(book, topic, subtopic)
print("Selected units:", selected_units)
print("Relevant text:\n", relevant_text)
```