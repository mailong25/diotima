# Table-of-Content Approach

Build structured knowledge from textbooks/PDFs — extract a Table of Contents, slice out section text, summarize subtopics, map curriculum learning outcomes to relevant textbook subtopics, and finally extract only the text that supports each outcome.

---

## What’s inside

```
repo/
├─ book_text_extraction.py             # Parse TOC pages → JSON outline + subtopic summaries
├─ automatic_subtopic_mapping.py       # Map curriculum outcomes → textbook subtopics (LLM + embeddings)
├─ subtopic_relevant_extraction.py     # Extract only text relevant to each outcome
├─ prompts/
│  └─ rag_prompt.yaml                  # All LLM prompt templates
├─ utils/
│  ├─ api_clients.py                   # openai_chat(prompt, model, max_tokens=...) expected
│  ├─ pdf_utils.py                     # pdf_to_text(pdf_path, page_start, page_end)
│  ├─ text_processing.py               # get_last_index(text, substring), text_to_chunks(...)
│  └─ prompt_processing.py             # get_prompt_template(yaml_path, key)
└─ config.py                           # Model name constants used by the scripts
```

---

## Quick start

### 1) Environment

- Python 3.10+ recommended

Install deps:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, a minimal set is:

```bash
pip install sentence-transformers tqdm pyyaml regex
```

> You’ll also need whatever your `utils/api_clients.py` uses (e.g., `openai`, `anthropic`, etc.).

### 2) Configure models & API keys

Edit **`config.py`** to point to the models you want:

```python
SECTION_SUMMARIZATION_MODEL = "gpt-4o-mini"              # example
TOC_EXTRACTION_MODEL = "gpt-4o-mini"                     # example
SUBTOPIC_SELECTION_MODEL = "gpt-4o"                      # example
SUBTOPIC_RELEVANT_TEXT_EXTRACTION_MODEL = "gpt-4o"       # example
```
Make sure **`utils/api_clients.openai_chat`** is implemented and has access to your API key(s), defined on the config file (OPENAI_API_KEY)

### 3) Verify prompts

`prompts/rag_prompt.yaml` contains all templates used by the scripts. Don’t change the keys unless you also update the code.

---

## End-to-end pipeline

### A) Extract book structure & subtopic summaries

**Input:** a textbook **PDF** and the page range of its **Table of Contents** (TOC)  
**Output:** a JSON outline with topics, subtopics, page ranges, raw text per subtopic, and a short **summary** for each subtopic.

**Run:**

```bash
python book_text_extraction.py   --pdf_path /path/to/book.pdf   --toc_start 5   --toc_end 12   --page_offset 12   --output_path data/book_outline.json
```

**Parameters explained**

- `--toc_start`, `--toc_end`: **actual PDF page indices** (0-based/1-based depends on your `pdf_to_text` — match whatever it expects).
- `--page_offset`: Offset to map TOC page numbers to actual PDF pages.  
  Example: if TOC “page 1” corresponds to PDF page 13, use `--page_offset=12`.

**What it does**

1. Extracts TOC text (`pdf_to_text`).
2. Calls **TOC LLM** (`toc_extraction`) to return JSON of chapters/subtopics with `page_start`, `page_end`.  
   > The code expects the LLM to return a fenced JSON block like:
   >
   > ```json
   > [ { "name": "...", "subtopics": [ { "name": "...", "page_start": 1, "page_end": 5 } ] } ]
   > ```
3. Normalizes page indices, extracts each subtopic’s **raw text** (`pdf_to_text`).
4. Removes trailing “chapter summary” if present (via `get_last_index`).
5. Summarizes each subtopic with **`subtopic_summary`** and writes the final JSON.

**Output schema (simplified)**

```json
[
  {
    "name": "Chapter Title",
    "page_start": 13,
    "page_end": 45,
    "subtopics": [
      {
        "name": "Subtopic Title",
        "page_start": 14,
        "page_end": 20,
        "text": "Raw extracted text...",
        "summary": "One-paragraph summary..."
      }
    ]
  }
]
```

---

### B) Map curriculum outcomes → textbook subtopics

**Input:** one or more book JSON(s) from step A + a curriculum CSV  
**Output:** a new CSV with an added `textbook_subtopics` column listing the chosen subtopics per outcome.

**Expected CSV columns**

Your curriculum CSV **must** include:

- `cur_topic`
- `cur_subtopic`
- `cur_outcome`  *(the learning outcome text)*

**Run:**

```bash
python automatic_subtopic_mapping.py   --book_path data/book_outline.json   --mapping_filepath data/curriculum.csv   --top_k 10   --output_mapping_csv data/curriculum_mapped.csv
```

**Optional: choose an embedding model**

```bash
--embedding_model "Qwen/Qwen3-Embedding-4B"
```

(Defaults to that value.)

**What it does**

1. Loads all subtopic **summaries** from the book JSON(s) and builds chunks like  
   `"{subtopic_name} -- {subtopic_summary}"`.
2. Embeds all chunks (`sentence-transformers`).
3. For each `cur_outcome`, embeds the query and retrieves **top-k** most similar chunks.
4. Sends those candidate sections to the LLM (`subtopic_mapping` prompt) to pick **only** the titles that directly address the outcome.
5. Writes `textbook_subtopics` in the output CSV (pipe-separated if multiple).

**Output CSV (new column)**

- `textbook_subtopics` — e.g. `Measurements | Units and Accuracy`

---

### C) Extract only the text that supports each outcome

**Input:** book JSON(s) + the mapped CSV from step B  
**Output:** a **cleaned book JSON** (topic → subtopic → extracted text) keeping only text relevant to each learning outcome.

**Run:**

```bash
python subtopic_relevant_extraction.py   --book_path data/book_outline.json   --mapped_curriculum data/curriculum_mapped.csv   --clean_book_path data/book_clean.json
```

**What it does**

1. For every `cur_subtopic` in your CSV, finds the mapped `textbook_subtopics` in the book JSON(s) and concatenates their raw text (with headers).
2. Calls the LLM with the `subtopic_text_extraction` prompt to **extract only** the minimal text that supports the **learning outcome** (`cur_outcome`).
   - Works in chunks if text > 4000 words (`text_to_chunks`, `CHUNK_MAX_WORDS=4000`).
3. If nothing relevant is found, the entry is **dropped** for that `cur_subtopic`.
4. Writes an incremental `book_clean.json` as it progresses.

**Output (structure)**

```json
{
  "Topic A": {
    "Subtopic 1": "Only the extracted text that supports the outcome...",
    "Subtopic 2": "..."
  },
  "Topic B": { ... }
}
```

---

## Example end-to-end commands

```bash
# A) Outline + summaries from PDF
python book_text_extraction.py   --pdf_path ./books/physics.pdf   --toc_start 8   --toc_end 16   --page_offset 10   --output_path ./out/physics_outline.json

# B) Map curriculum outcomes to subtopics
python automatic_subtopic_mapping.py   --book_path ./out/physics_outline.json   --mapping_filepath ./curriculum/leaving_cert_physics.csv   --top_k 12   --output_mapping_csv ./out/physics_curriculum_mapped.csv

# C) Extract only text relevant to each outcome
python subtopic_relevant_extraction.py   --book_path ./out/physics_outline.json   --mapped_curriculum ./out/physics_curriculum_mapped.csv   --clean_book_path ./out/physics_clean.json
```

---
