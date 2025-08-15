import os
import json
import argparse
from utils.text_processing import text_to_chunks
from utils.api_clients import openai_chat
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SUBTOPIC_RELEVANT_TEXT_EXTRACTION_MODEL
from utils.prompt_processing import get_prompt_template
from collections import defaultdict
import time

def subtopic_text_extraction(text, subtopic):
    try:
        CHUNK_MAX_WORDS = 2000
        def _get_response(text_chunk, subtopic):
            prompt_template = get_prompt_template("prompts/chunk_retrieval.yaml", "subtopic_text_extraction")
            prompt = prompt_template.format(chunk=text_chunk, subtopic=subtopic)
            response = openai_chat(prompt, model=SUBTOPIC_RELEVANT_TEXT_EXTRACTION_MODEL, max_tokens=16384).strip()
            match = re.search(r'Extracted text:\s*(.*)', response, re.DOTALL)
            if match:
                response = match.group(1).strip()
            else:
                response = "NONE"
            return response if "NONE" not in response else ""
        
        word_count = len(text.split())
        
        if word_count <= CHUNK_MAX_WORDS:
            result = _get_response(text, subtopic)
        else:
            chunks = text_to_chunks(text, chunk_max_words=CHUNK_MAX_WORDS, splitter=".\n")
            responses = [_get_response(chunk, subtopic) for chunk in chunks]
            result = '.\n\n'.join(responses).strip()
        
        return result if len(result) > 10 else None
    except Exception as e:
        print(f"Error during unit knowledge extraction: {e}")
        print("Returning original text:", subtopic)
        return text

def main():
    parser = argparse.ArgumentParser(description='Process RAG book data with subtopic extraction')
    parser.add_argument('--book_path', nargs="+", help='Path to the book JSON file, can be multiple. E.g. resources/bio_book_raw.json')
    parser.add_argument('--curriculum_csv', required=True, help='Path to mapped book. E.g. resources/curriculum_mapping.csv . The mapping might not be perfect, so expert review is recommended.')
    parser.add_argument('--condensed_output_path', required=True, help='Path to output book. E.g. resources/structured_book_toc.json')
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use for summarization. Defaults to CPU count if omitted."
    )
    
    args = parser.parse_args()

    # Load data
    books = []
    for path in args.book_path:
        books += json.load(open(path))
    
    mapped_book = defaultdict(dict)    
    subtopic_to_outcome = {}

    with open(args.curriculum_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:

            cur_topic = row['cur_topic']
            cur_subtopic = row['cur_subtopic'].strip()
            cur_outcome = row['cur_outcome']
            mapped_subtopics = row['textbook_subtopics']
            subtopic_texts = []
            subtopic_to_outcome[cur_subtopic] = cur_outcome

            if len(mapped_subtopics) > 0:
                mapped_subtopics = mapped_subtopics.split(' | ')
                for mapped_subtopic in mapped_subtopics:
                    check = False
                    for topic in books:
                        for subtopic in topic['subtopics']:
                            if subtopic['name'].lower() == mapped_subtopic.lower():
                                header = f"\n-------{subtopic['name']}-------\n"
                                subtopic_texts.extend([header, subtopic['text']])
                                check = True
                    if not check:
                        print(f"Warning: No text found for mapped '{mapped_subtopic}'")
            
            if subtopic_texts:
                mapped_book[cur_topic][cur_subtopic] = '\n'.join(subtopic_texts)
            else:
                print(f"Warning: No mapping found for subtopic: '{cur_subtopic}'")

    def process_subtopic(args_tuple):
        topic, subtopic, text = args_tuple
        extracted_text = subtopic_text_extraction(text, subtopic_to_outcome[subtopic])
        return topic, subtopic, extracted_text

    # Collect all tasks in order
    mapped_book = dict(mapped_book)
    tasks = []
    for topic in mapped_book:
        for subtopic in mapped_book[topic]:
            text = mapped_book[topic][subtopic]
            tasks.append((topic, subtopic, text))

    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        # Process tasks in order using executor.map()
        results = executor.map(process_subtopic, tasks)
        
        # Process results in the same order they were submitted
        for topic, subtopic, extracted_text in results:
            if extracted_text is None:
                del mapped_book[topic][subtopic]
            else:
                mapped_book[topic][subtopic] = extracted_text
            
            json.dump(mapped_book, open(args.condensed_output_path, 'w'))

if __name__ == "__main__":
    main()