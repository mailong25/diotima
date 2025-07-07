

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
# Extract content of the first book
python build_book_json.py \
  --pdf_path /path/to/ap_book.pdf \
  --tab_start 6 \
  --tab_end 9 \
  --page_offset 9 \
  --output_path resources/ap_book.json
```

## Arguments
```
--pdf_path	Path to the PDF file.
--tab_start	Start page index (in the PDF) of the table of contents.
--tab_end	End page index (in the PDF) of the table of contents.
--page_offset	Offset to map the TOC page numbers to the actual PDF page indices (e.g., if TOC page 1 is PDF 13).
--output_path	Path to save the generated book.json file.
```

## Output json format
```
[
	{
	    "name": "topic #1",
	    "subtopics": [
	      {
		"name": "subtopic #1",
		"page_start": 17,
		"page_end": 29,
	  	"text": "Sample text"
	      }
 	      {
        	"name": "subtopic #2",
        	"page_start": 29,
        	"page_end": 49,
        	"text": "sample_text"
              }
	      ...
        }
        ,
	{
	    "name": "topic #2",
        }
]
```


# Megre book content in case there are multiple books
```
python merge_book.py resources/ap_book.json,resources/lab_book.json resources/merged_book.json
```

# Map humman annotation results
```
python subtopic_mapping.py --book_path resources/merged_book.json --mapping_path resources/mapping.csv --output_path resources/mapped_curriculum.json
```

## 🔍 2️⃣ Extract subtopic text from the processed book
🔧 Example Usage
```
def find_subtopic_text(mapped_curriculum, subtopic_name):
    """
    Search for the subtopic_name in the mapped_curriculum and return its text content.
    """
    for topic, subtopics in mapped_curriculum.items():
        if subtopic_name in subtopics:
            return subtopics[subtopic_name]
    print(f"Warning: Subtopic '{subtopic_name}' not found in mapped curriculum.")
    return None

mapped_curriculum = json.load(open("resources/mapped_curriculum.json"))
print(find_subtopic_text(mapped_curriculum, "Scientists and scientific ideas"))
```
