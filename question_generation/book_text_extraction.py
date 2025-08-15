#!/usr/bin/env python3
import argparse
import json
from utils.api_clients import openai_chat
from utils.pdf_utils import pdf_to_text
from utils.text_processing import get_last_index
from utils.prompt_processing import get_prompt_template
from tqdm import tqdm
from config import SECTION_SUMMARIZATION_MODEL, TOC_EXTRACTION_MODEL
from multiprocessing import Pool

def subtopic_summary(text, main_section, leading_section, trailing_section):
    prompt_template = get_prompt_template("prompts/chunk_retrieval.yaml", "subtopic_summarization")

    prompt = prompt_template.format(
        text=text,
        main_section=main_section,
        leading_section=leading_section,
        trailing_section=trailing_section
    )
    response = openai_chat(prompt, model=SECTION_SUMMARIZATION_MODEL, max_tokens=10000).strip()
    return response

def toc_extraction(text):
    prompt_template = get_prompt_template("prompts/chunk_retrieval.yaml", "toc_extraction")
    prompt = prompt_template.format(toc_content=text)
    resp = openai_chat(prompt, model=TOC_EXTRACTION_MODEL)

    json_str = resp.split("```json\n")[1].split("```")[0].strip()
    toc = json.loads(json_str)
    return toc

def _build_tasks_from_toc(toc):
    """
    Flatten the toc into a global list of subtopic tasks with deterministic
    prev/next context, along with their (i, j) indices to write back.
    """
    flat = []
    for i, topic in enumerate(toc):
        for j, sub in enumerate(topic["subtopics"]):
            flat.append(((i, j), sub["name"], sub["text"]))

    tasks = []
    for k, ((i, j), name, text) in enumerate(flat):
        prev_name = flat[k - 1][1] if k - 1 >= 0 else "None"
        next_name = flat[k + 1][1] if k + 1 < len(flat) else "None"
        tasks.append({
            "i": i,
            "j": j,
            "name": name,
            "text": text,
            "prev": prev_name,
            "next": next_name
        })
    return tasks

def _summary_worker(task):
    try:
        summary = subtopic_summary(
            text=task["text"],
            main_section=task["name"],
            leading_section=task["prev"],
            trailing_section=task["next"]
        )
        return (task["i"], task["j"], summary)
    except Exception as e:
        print(f"Pipeline summary failed: {e}")
        raise

def main(pdf_path, toc_start, toc_end, page_offset, output_path, parallel_workers):
    toc_raw_text = pdf_to_text(pdf_path, toc_start, toc_end)

    try:
        toc = toc_extraction(toc_raw_text)
    except Exception as e:
        print("Error parsing JSON from LLM response:", e)
        return

    # Normalize page indices and extract text for each subtopic
    for topic in toc:
        topic['subtopics'][0]['page_start'] -= 1

        for subtopic in topic['subtopics']:
            subtopic['page_start'] += page_offset
            subtopic['page_end'] += page_offset + 1
            text = pdf_to_text(pdf_path, subtopic['page_start'], subtopic['page_end'])
            to_remove_idx = get_last_index(text, "chapter summary")
            if to_remove_idx != -1:
                text = text[:to_remove_idx]
            subtopic['text'] = text

        topic['page_start'] = topic['subtopics'][0]['page_start']
        topic['page_end'] = topic['subtopics'][-1]['page_end']

    # Build tasks with deterministic prev/next context across the entire book
    tasks = _build_tasks_from_toc(toc)
    
    print("Generating subtopic summaries...")

    # Run in parallel with a progress bar
    with Pool(processes=parallel_workers) as pool:
        results_iter = pool.imap_unordered(_summary_worker, tasks)
        results = []
        for res in tqdm(results_iter, total=len(tasks)):
            results.append(res)

    # Write summaries back into the original nested structure
    for i, j, summary in results:
        toc[i]['subtopics'][j]['summary'] = summary

    with open(output_path, "w") as f:
        json.dump(toc, f, indent=2)

    print(f"Outline successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Build book JSON file from a PDF.
        Notes:
        - The --page_offset parameter is the offset to map TOC page numbers to the actual PDF page indices.
          For example, if TOC says page 1 is actually page 13 in PDF, then --page_offset=12.
        """
    )
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file. E.g. resources/bio_book.pdf")
    parser.add_argument("--toc_start", type=int, required=True, help="Start page of the table of contents (actual PDF page index). E.g. 6")
    parser.add_argument("--toc_end", type=int, required=True, help="End page of the table of contents (actual PDF page index). E.g. 9")
    parser.add_argument("--page_offset", type=int, required=True, help="Page offset to map TOC page numbers to actual PDF page numbers. E.g. 9")
    parser.add_argument("--output_path", required=True, help="Path to save the output JSON file. E.g. resources/bio_book_raw.json")
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use for summarization. Defaults to CPU count if omitted."
    )
    args = parser.parse_args()

    main(
        pdf_path=args.pdf_path,
        toc_start=args.toc_start,
        toc_end=args.toc_end,
        page_offset=args.page_offset,
        output_path=args.output_path,
        parallel_workers=args.parallel_workers
    )