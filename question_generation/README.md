# 📚 Curriculum ↔ Textbook Mapping Pipelines

This repo contains two complementary pipelines for turning a textbook PDF into **outcome-aligned** learning materials.

- **Table-of-Contents (TOC) approach**: parse the book’s table of contents, pull the exact pages per subtopic, summarize them, map them to your curriculum, then extract only the text relevant to each learning outcome.
- **RAG-based approach**: extract structured knowledge from the whole book and rank it for each learning outcome using embeddings + an LLM relevance check.

Both pipelines rely on the same config and utils, but produce different artifacts. Pick the one that fits your data—or run both and compare.

---

## 🧰 Prerequisites

### 1) Python & OS
- Python **3.9–3.11** recommended  
- macOS / Linux / Windows

### 2) Download the book
```bash
Download biology book from: https://openstax.org/details/books/biology-ap-courses
Name it as your_book.pdf and put it into resources/your_book.pdf
```

### 3) API keys & config
Set keys as environment variables (read by `config.py`):
```bash
export OPENAI_API_KEY="sk-..."      # required
export MISTRAL_API_KEY="mistral-..." # optional (only if you wire mistral in)
```

Model names are set in **`config.py`**:
```python
KNOWLEDGE_EXTRACTION_MODEL = "gpt-4.1"
RELEVANCE_CHECK_MODEL = "gpt-4.1-mini"
SECTION_SUMMARIZATION_MODEL = "gpt-4.1"
TOC_EXTRACTION_MODEL = "gpt-4o"
SUBTOPIC_SELECTION_MODEL = "gpt-5-chat-latest"
SUBTOPIC_RELEVANT_TEXT_EXTRACTION_MODEL = "gpt-4.1"
QUERY_EXPANSION_MODEL = "gpt-5-chat-latest"
```
> If a model isn’t available to your account, change it to one you can access.

### 4) Project layout (expected)
```
.
├─ config.py
├─ utils/
│  ├─ api_clients.py
│  ├─ pdf_utils.py          # must provide pdf_to_text(pdf_path, start, end)
│  ├─ text_processing.py    # get_last_index, text_to_chunks
│  └─ prompt_processing.py  # get_prompt_template
├─ prompts/
│  └─ chunk_retrieval.yaml
├─ resources/
│  ├─ your_book.pdf
│  └─ curriculum.csv
├─ book_text_extraction.py
├─ automatic_subtopic_mapping.py
├─ subtopic_relevant_extraction.py
├─ rag_extract_knowledge.py
└─ rag_relevance_ranking.py
```

### 5) Curriculum CSV schema
Your CSV **must** have:
- `cur_topic` — e.g., “Cell Biology”
- `cur_subtopic` — e.g., “Cell membrane structure”
- `cur_outcome` — the learning outcome (used as the query)

Example:
```csv
cur_topic,cur_subtopic,cur_outcome
Cell Biology,Cell membrane structure,Describe the fluid mosaic model of membranes
Genetics,Mendelian inheritance,Explain segregation and independent assortment
```

---

## 🔀 Choose a Pipeline

- **TOC approach** → precise page ranges; best when TOC is clean.  
- **RAG approach** → ignores TOC; retrieves and ranks knowledge chunks from any page span.

---

# 1) 🧭 Table-of-Contents Approach

**Flow:** `book_text_extraction.py` → `automatic_subtopic_mapping.py` → `subtopic_relevant_extraction.py`


## Step 1 — Parse TOC, extract pages, and summarize each subtopic

This script:
- Reads TOC pages and asks an LLM to produce a machine-readable TOC.
- Normalizes page numbers using `--page_offset`.
- Extracts **raw text** for each subtopic range.
- Summarizes each subtopic with `SECTION_SUMMARIZATION_MODEL`.
- Saves a JSON with `[chapter → subtopics → {page_start, page_end, text, summary}]`.

**Run**
```bash
python book_text_extraction.py   --pdf_path resources/your_book.pdf   --toc_start 6 --toc_end 9   --page_offset 12   --output_path resources/book_raw.json   --parallel_workers 4
```

**Example: TOC snippet read from PDF**
```
Table of Contents
1 Introduction ........................................ 1
  1.1 What is life? ................................... 2
  1.2 Scientific method ............................... 6
2 Cell Biology ........................................ 13
  2.1 Cell membrane ................................... 14
  2.2 Transport across membranes ...................... 21
```

**Example Output (`resources/book_raw.json`, snippet)**
```json
[
  {
    "name": "Introduction",
    "page_start": 13,
    "page_end": 25,
    "subtopics": [
      {
        "name": "What is life?",
        "page_start": 14,
        "page_end": 18,
        "text": "Life is characterized by organization, metabolism, homeostasis, growth, ...",
        "summary": "Defines life via shared characteristics and introduces levels of biological organization."
      },
      {
        "name": "Scientific method",
        "page_start": 19,
        "page_end": 25,
        "text": "The scientific method involves observation, hypothesis formation, prediction, ...",
        "summary": "Outlines steps of the scientific method and the role of falsifiability and replication."
      }
    ]
  },
  {
    "name": "Cell Biology",
    "page_start": 26,
    "page_end": 41,
    "subtopics": [
      {
        "name": "Cell membrane",
        "page_start": 26,
        "page_end": 33,
        "text": "Membranes follow the fluid mosaic model with phospholipid bilayers and embedded proteins...",
        "summary": "Explains membrane composition, fluidity factors, and protein roles."
      },
      {
        "name": "Transport across membranes",
        "page_start": 34,
        "page_end": 41,
        "text": "Diffusion, osmosis, facilitated diffusion via channels/carriers, and active transport using ATP...",
        "summary": "Compares passive vs active transport with examples (ion pumps, aquaporins)."
      }
    ]
  }
]
```
> ⚠️ The script trims trailing “chapter summary” if it finds that phrase. Adjust `get_last_index` if your book uses different headings.


## Step 2 — Map curriculum outcomes to the most relevant subtopics

This script:
- Builds embeddings for **subtopic summaries** from Step 1.
- Retrieves top-K candidate sections for each learning outcome and asks an LLM which ones are truly relevant.
- Writes a **new CSV** with a `textbook_subtopics` column listing chosen sections.

**Run**
```bash
python automatic_subtopic_mapping.py   --book_path resources/book_raw.json   --curriculum_csv resources/curriculum.csv   --output_mapping_csv resources/curriculum_mapping.csv   --top_k 10   --embedding_model Qwen/Qwen3-Embedding-4B
```

**Example Input CSV (`resources/curriculum.csv`, snippet)**
```csv
cur_topic,cur_subtopic,cur_outcome
Cell Biology,Cell membrane structure,Describe the fluid mosaic model and factors that affect membrane fluidity
Cell Biology,Membrane transport,Compare diffusion, osmosis, facilitated diffusion, and active transport
```

**Example Output CSV (`resources/curriculum_mapping.csv`, snippet)**
```csv
cur_topic,cur_subtopic,cur_outcome,textbook_subtopics
Cell Biology,Cell membrane structure,"Describe the fluid mosaic model and factors that affect membrane fluidity","Cell membrane"
Cell Biology,Membrane transport,"Compare diffusion, osmosis, facilitated diffusion, and active transport","Transport across membranes | Cell membrane"
```
> Note: `automatic_subtopic_mapping.py` sets `CUDA_VISIBLE_DEVICES=-1` (CPU only). Remove/change that line for GPU.


## Step 3 — Extract only the text that directly supports each learning outcome

This script:
- Gathers the **raw subtopic texts** (not summaries) for the sections chosen in Step 2.
- Sends them to the LLM to **extract only the material directly relevant** to each outcome.
- Saves a condensed JSON keyed by `cur_topic → cur_subtopic`.

**Run**
```bash
python subtopic_relevant_extraction.py   --book_path resources/book_raw.json   --curriculum_csv resources/curriculum_mapping.csv   --condensed_output_path resources/structured_book_toc.json   --parallel_workers 6
```

**Conceptual input assembled internally**
```
-------Cell membrane-------
Membranes follow the fluid mosaic model with a phospholipid bilayer...
Cholesterol modulates fluidity by preventing tight packing at low temp...

-------Transport across membranes-------
Diffusion is movement down a concentration gradient...
Active transport requires energy, often ATP, via pumps such as Na+/K+ ATPase...
```

**Example Output (`resources/structured_book_toc.json`, snippet)**
```json
{
  "Cell Biology": {
    "Cell membrane structure": "Membranes follow the fluid mosaic model with a phospholipid bilayer and proteins that move laterally. Cholesterol modulates fluidity by preventing tight packing at low temperature and restraining movement at high temperature. Unsaturated fatty acids increase fluidity; saturated fatty acids decrease it.",
    "Membrane transport": "Diffusion moves molecules down a concentration gradient without energy input. Osmosis is water diffusion across a selectively permeable membrane. Facilitated diffusion uses channels or carriers but remains passive. Active transport moves solutes against gradients using energy (e.g., Na+/K+ ATPase)."
  }
}
```

---

# 2) 🔎 RAG-Based Approach

**Flow:** `rag_extract_knowledge.py` → `rag_relevance_ranking.py`


## Step 1 — Convert PDF pages into structured knowledge chunks

This script:
- Extracts text from a page range.
- Chunks it (~2000 words max per chunk).
- Uses an LLM to **extract factual knowledge** and **paragraph it**.

**Run**
```bash
python rag_extract_knowledge.py   --source-pdf resources/your_book.pdf   --start-page 18   --end-page 120   --parallel-workers 8   --output-knowledge-file resources/book_knowledge.txt
```

**Example Output (`resources/book_knowledge.txt`, snippet)**
```
Membranes consist of a phospholipid bilayer with hydrophilic heads and hydrophobic tails. Proteins embedded within the bilayer serve transport, signaling, and structural roles. Cholesterol intercalates among phospholipids and modulates fluidity. Temperature and lipid saturation influence membrane viscosity.

Diffusion is the net movement of molecules from high to low concentration due to random motion. Osmosis refers specifically to water diffusion across a selectively permeable membrane. Facilitated diffusion requires specific membrane proteins but does not consume metabolic energy. Active transport uses energy, typically ATP hydrolysis, to move solutes against their gradients, exemplified by the Na⁺/K⁺-ATPase.
```

## Step 2 — Rank knowledge chunks for each learning outcome (RAG)

This script:
- Loads `book_knowledge.txt` (or multiple files).
- For each outcome, generates related queries (query expansion).
- Retrieves top-K similar chunks via embeddings.
- Asks an LLM to label each candidate as **Not / Low / Medium / High / Very high relevance**.
- Saves both a **detailed mapping** and a **condensed knowledge base** (high-relevance only).

**Run**
```bash
python rag_relevance_ranking.py   --source-files resources/book_knowledge.txt   --curriculum-csv resources/curriculum.csv   --embedding-model Qwen/Qwen3-Embedding-4B   --top-k 10   --detailed-output-path resources/detailed_rag_mapping.json   --condensed-output-path resources/structured_book_rag.json
```

**Example Detailed Output (`resources/detailed_rag_mapping.json`, snippet)**
```json
{
  "Cell Biology": {
    "Cell membrane structure": [
      {
        "content": "Membranes consist of a phospholipid bilayer with hydrophilic heads ...",
        "relevance_level": "Very high relevance",
        "source_query": "Describe the fluid mosaic model and factors that affect membrane fluidity",
        "chunk_index": 0
      },
      {
        "content": "Temperature and lipid saturation influence membrane viscosity ...",
        "relevance_level": "High relevance",
        "source_query": "Describe the fluid mosaic model and factors that affect membrane fluidity",
        "chunk_index": 0
      }
    ],
    "Membrane transport": [
      {
        "content": "Diffusion is the net movement of molecules from high to low concentration ...",
        "relevance_level": "Very high relevance",
        "source_query": "Compare diffusion, osmosis, facilitated diffusion, and active transport",
        "chunk_index": 1
      }
    ]
  }
}
```

**Example Condensed Output (`resources/structured_book_rag.json`, snippet)**
```json
{
  "Cell Biology": {
    "Cell membrane structure": "Membranes consist of a phospholipid bilayer with hydrophilic heads and hydrophobic tails. Proteins embedded within the bilayer serve transport, signaling, and structural roles. Cholesterol intercalates among phospholipids and modulates fluidity. Temperature and lipid saturation influence membrane viscosity.",
    "Membrane transport": "Diffusion is the net movement of molecules from high to low concentration due to random motion. Osmosis refers specifically to water diffusion across a selectively permeable membrane. Facilitated diffusion requires specific membrane proteins but does not consume metabolic energy. Active transport uses energy, typically ATP hydrolysis, to move solutes against their gradients, exemplified by the Na⁺/K⁺-ATPase."
  }
}
```

---

## ✅ Quick Start (copy/paste)

**TOC pipeline**
```bash
# 1) TOC → subtopic texts + summaries
python book_text_extraction.py   --pdf_path resources/your_book.pdf   --toc_start 6 --toc_end 9 --page_offset 12   --output_path resources/book_raw.json   --parallel_workers 4

# 2) Map outcomes → subtopics
python automatic_subtopic_mapping.py   --book_path resources/book_raw.json   --curriculum_csv resources/curriculum.csv   --output_mapping_csv resources/curriculum_mapping.csv

# 3) Extract outcome-relevant text
python subtopic_relevant_extraction.py   --book_path resources/book_raw.json   --curriculum_csv resources/curriculum_mapping.csv   --condensed_output_path resources/structured_book_toc.json   --parallel_workers 6
```

**RAG pipeline**
```bash
# 1) Extract structured knowledge from PDF (remove the 5-chunk limiter in code for full runs)
python rag_extract_knowledge.py   --source-pdf resources/your_book.pdf   --start-page 18 --end-page 1770   --parallel-workers 8   --output-knowledge-file resources/book_knowledge.txt

# 2) Rank relevance per outcome
python rag_relevance_ranking.py   --source-files resources/book_knowledge.txt   --curriculum-csv resources/curriculum.csv   --detailed-output-path resources/detailed_rag_mapping.json   --condensed-output-path resources/structured_book_rag.json
```

---

## 🧪 Copy-ready mini fixtures (for quick testing)

Place in `resources/`:

**`curriculum.csv`**
```csv
cur_topic,cur_subtopic,cur_outcome
Cell Biology,Cell membrane structure,Describe the fluid mosaic model and factors that affect membrane fluidity
Cell Biology,Membrane transport,Compare diffusion, osmosis, facilitated diffusion, and active transport
```

**`book_knowledge.txt`**
```
Membranes consist of a phospholipid bilayer with hydrophilic heads and hydrophobic tails. Proteins embedded within the bilayer serve transport, signaling, and structural roles. Cholesterol intercalates among phospholipids and modulates fluidity. Temperature and lipid saturation influence membrane viscosity.

Diffusion is the net movement of molecules from high to low concentration due to random motion. Osmosis refers specifically to water diffusion across a selectively permeable membrane. Facilitated diffusion requires specific membrane proteins but does not consume metabolic energy. Active transport uses energy, typically ATP hydrolysis, to move solutes against their gradients, exemplified by the Na⁺/K⁺-ATPase.
```

---

## 📝 Tips & Troubleshooting

- **TOC JSON parsing failed** in `book_text_extraction.py`  
  Ensure `--toc_start/--toc_end` pages contain only the TOC and provide enough context. If the TOC is unusual, try fewer pages or edit the TOC pages.

- **Wrong page ranges**  
  Double-check `--page_offset`. It assumes a constant offset from TOC numbering to actual PDF indices.

- **Embedding model memory**  
  `Qwen/Qwen3-Embedding-4B` is strong but heavier. Swap for a lighter SentenceTransformers model if you hit memory limits.

- **GPU usage**  
  `automatic_subtopic_mapping.py` disables GPU via `CUDA_VISIBLE_DEVICES="-1"`. Remove that line for GPU. `rag_relevance_ranking.py` uses CUDA if available.

- **Token / cost control**  
  Lower `--top_k`, reduce page ranges, or switch to smaller LLMs in `config.py`.

- **Chunking**  
  Default ~2000 words per chunk. Adjust in code if your PDFs produce overly long/short chunks.

- **PDF text quality**  
  If `pdf_to_text` yields poor text (tables, formulas), consider improving extraction in `utils/pdf_utils.py` (e.g., `pdfminer.six`, OCR).

---

## 📦 Outputs at a Glance

- **TOC pipeline**
  - `book_raw.json` → per-subtopic `{ text, summary, page_start, page_end }`
  - `curriculum_mapping.csv` → original CSV + `textbook_subtopics`
  - `structured_book_toc.json` → distilled, outcome-aligned excerpts

- **RAG pipeline**
  - `book_knowledge.txt` → factual, paragraph-organized chunks
  - `detailed_rag_mapping.json` → all candidate chunks with relevance labels
  - `structured_book_rag.json` → high-relevance only, merged per subtopic

---