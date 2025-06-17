#!/usr/bin/env python3

import argparse
import json
import yaml
from langchain.prompts import PromptTemplate
from utils.api_clients import openai_chat
from utils.pdf_utils import pdf_to_text
from utils.text_processing import text_to_chunks
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import UNIT_KNOWLEDGE_EXTRACTION_MODEL

def unit_knowledge_extraction(text):
    try:
        with open("prompts/rag_prompt.yaml", "r") as f:
            templates = yaml.safe_load(f)

        template_data = templates['unit_knowledge_extraction']
        prompt_template = PromptTemplate(
            input_variables=template_data["input_variables"],
            template=template_data["template"]
        )

        CHUNK_MAX_WORDS = 2000

        word_count = len(text.split())
        if word_count <= CHUNK_MAX_WORDS:
            prompt = prompt_template.format(text=text)
            response = openai_chat(prompt, model=UNIT_KNOWLEDGE_EXTRACTION_MODEL, max_tokens=16384)
            return response
        else:
            chunks = text_to_chunks(text, chunk_max_words=CHUNK_MAX_WORDS, splitter=".\n")
            responses = []
            
            for i, chunk in enumerate(chunks):
                prompt = prompt_template.format(text=chunk)
                response = openai_chat(prompt, model=UNIT_KNOWLEDGE_EXTRACTION_MODEL, max_tokens=16384)
                responses.append(response)
            
            # Concatenate all responses
            return '.\n'.join(responses)
    except:
        print("Error during unit knowledge extraction. Returning original text.")
        return text

def process_subtopic(subtopic_data):
    subtopic_data['text'] = unit_knowledge_extraction(subtopic_data['text'])
    return subtopic_data

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

    print("Calling the LLM to get the outline...")
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

            # Prepend topic name into subtopic
            subtopic['name'] = subtopic['name'].replace(':', '-')
            subtopic['name'] = topic['name'] + ': ' + subtopic['name']

        topic['page_start'] = topic['subtopics'][0]['page_start']
        topic['page_end'] = topic['subtopics'][-1]['page_end']

    # Collect all subtopics for parallel processing
    all_subtopics = []
    for topic in outline:
        all_subtopics.extend(topic['subtopics'])
    
    # Process all subtopics in parallel
    print("Processing subtopics with unit knowledge extraction...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_subtopic = {
            executor.submit(process_subtopic, subtopic): subtopic 
            for subtopic in all_subtopics
        }
        
        for future in tqdm(as_completed(future_to_subtopic), total=len(all_subtopics), desc="Extracting knowledge"):
            try:
                processed_subtopic = future.result()
            except Exception as e:
                subtopic = future_to_subtopic[future]
                print(f"Error processing subtopic: {e}")

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
