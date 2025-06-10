#!/usr/bin/env python3

import argparse
import json
import yaml
from langchain.prompts import PromptTemplate
from utils.api_clients import openai_chat
from utils.pdf_utils import pdf_to_text

def main(pdf_path, tab_start, tab_end, page_offset, output_path):
    # Extract table of contents text
    tab_content = pdf_to_text(pdf_path, tab_start, tab_end)

    # Load the prompt template
    with open("prompts/rag_prompt.yaml", "r") as f:
        templates = yaml.safe_load(f)
    template_data = templates['outline_extraction']

    # Prepare the prompt
    prompt_template = PromptTemplate(
        input_variables=template_data["input_variables"],
        template=template_data["template"]
    )
    prompt = prompt_template.format(tab_content=tab_content)

    # Call the LLM to get the outline
    resp = openai_chat(prompt)

    # Parse the JSON response
    try:
        json_str = resp.split("```json\n")[1].split("```")[0].strip()
        outline = json.loads(json_str)
    except Exception as e:
        print("Error parsing JSON from LLM response:", e)
        return

    # Adjust page numbers to map TOC to actual PDF page numbers
    for topic in outline:
        topic['subtopics'][0]['page_start'] -= 1
        topic['subtopics'][-1]['page_end'] -= 1

        for subtopic in topic['subtopics']:
            # Map TOC page to actual PDF page
            subtopic['page_start'] += page_offset
            subtopic['page_end'] += page_offset
            text = pdf_to_text(pdf_path, subtopic['page_start'], subtopic['page_end'])
            subtopic['text'] = text

        topic['page_start'] = topic['subtopics'][0]['page_start']
        topic['page_end'] = topic['subtopics'][-1]['page_end']

    # Save to output JSON
    with open(output_path, "w") as f:
        json.dump(outline, f, indent=2)

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
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file.")
    parser.add_argument("--tab_start", type=int, required=True, help="Start page of the table of contents (actual PDF page index).")
    parser.add_argument("--tab_end", type=int, required=True, help="End page of the table of contents (actual PDF page index).")
    parser.add_argument("--page_offset", type=int, required=True, help="Page offset to map TOC page numbers to actual PDF page numbers.")
    parser.add_argument("--output_path", required=True, help="Path to save the output JSON file.")
    args = parser.parse_args()

    main(
        pdf_path=args.pdf_path,
        tab_start=args.tab_start,
        tab_end=args.tab_end,
        page_offset=args.page_offset,
        output_path=args.output_path
    )
