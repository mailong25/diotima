#!/usr/bin/env python3

import argparse
from utils.api_clients import openai_chat
from utils.pdf_utils import pdf_to_text
from utils.text_processing import text_to_chunks
from multiprocessing import Pool
from tqdm import tqdm
from config import KNOWLEDGE_EXTRACTION_MODEL
import sys
from utils.prompt_processing import get_prompt_template

MAX_WORDS_PER_CHUNK = 2000

def extract_knowledge_from_text_chunk(text_chunk):
    """Process a single text chunk through LLM to extract structured knowledge."""
    try:
        prompt_template = get_prompt_template('prompts/chunk_retrieval.yaml','knowledge_extraction_and_paragraphing')
        formatted_prompt = prompt_template.format(text=text_chunk)
        
        llm_response = openai_chat(
            formatted_prompt, 
            model=KNOWLEDGE_EXTRACTION_MODEL, 
            max_tokens=16384
        ).strip()
        
        return llm_response.strip()
        
    except Exception as processing_error:
        print(f"Error processing text chunk: {processing_error}")
        return text_chunk  # Return original chunk if processing fails

def extract_knowledge_from_pdf_pages(source_pdf_path, start_page, end_page,
                                    output_knowledge_path, parallel_workers):
    """
    Main pipeline to extract structured knowledge from PDF pages.
    
    Args:
        source_pdf_path: Path to the source PDF file
        start_page: First page to process (inclusive)
        end_page: Last page to process (inclusive)
        output_knowledge_path: Where to save the extracted knowledge
        parallel_workers: Number of parallel processing workers
    """
    print("Extracting text content from PDF pages...")
    raw_pdf_text = pdf_to_text(source_pdf_path, start_page, end_page)
    
    print("Segmenting text into processable chunks...")
    text_chunks = text_to_chunks(
        raw_pdf_text, 
        chunk_max_words=MAX_WORDS_PER_CHUNK, 
        splitter=".\n"
    )
    
    print(f"{len(text_chunks)} text chunks created for knowledge extraction.")
    print("Processing chunks through LLM for knowledge extraction...")
    
    # Process chunks in parallel
    with Pool(processes=parallel_workers) as process_pool:
        extracted_knowledge_chunks = list(tqdm(
            process_pool.imap(extract_knowledge_from_text_chunk, text_chunks),
            total=len(text_chunks),
            desc="Extracting structured knowledge",
            unit="chunk"
        ))
    
    # Filter out chunks that are too short (likely failed extractions)
    MIN_WORDS_THRESHOLD = 10
    valid_knowledge_chunks = [
        knowledge_chunk for knowledge_chunk in extracted_knowledge_chunks 
        if len(knowledge_chunk.split()) > MIN_WORDS_THRESHOLD
    ]
    
    # Combine all extracted knowledge
    consolidated_knowledge = "\n\n".join(valid_knowledge_chunks)
    
    print("Saving consolidated knowledge to output file...")
    with open(output_knowledge_path, "w", encoding="utf-8") as output_file:
        output_file.write(consolidated_knowledge)

    print(f"Knowledge extraction completed successfully. Output saved to: {output_knowledge_path}")

if __name__ == "__main__":
    command_parser = argparse.ArgumentParser(
        description="""
        Extract and structure knowledge from specific PDF page ranges using LLM processing.

        Processing Pipeline:
        1. Extract raw text from the specified PDF page range
        2. Segment text into manageable chunks (max 3000 words, split on sentence boundaries)
        3. Process each chunk through LLM in parallel for knowledge extraction
        4. Consolidate and save the structured knowledge to a text file
        """
    )
    
    command_parser.add_argument(
        "--source-pdf", 
        required=True, 
        help="Path to the source PDF file to process. E.g. resources/bio_book.pdf"
    )
    command_parser.add_argument(
        "--start-page", 
        type=int, 
        required=True, 
        help="Starting page number for extraction (inclusive, 1-indexed). E.g. 18"
    )
    command_parser.add_argument(
        "--end-page", 
        type=int, 
        required=True, 
        help="Ending page number for extraction (inclusive, 1-indexed). E.g. 1770"
    )
    command_parser.add_argument(
        "--parallel-workers", 
        type=int, 
        required=True, 
        help="Number of parallel worker processes for LLM processing. E.g. 8"
    )
    command_parser.add_argument(
        "--output-knowledge-file", 
        required=True, 
        help="Path where the extracted knowledge will be saved (as .txt file). E.g. resources/bio_book_knowledge.txt"
    )
    
    script_arguments = command_parser.parse_args()

    extract_knowledge_from_pdf_pages(
        source_pdf_path=script_arguments.source_pdf,
        start_page=script_arguments.start_page,
        end_page=script_arguments.end_page,
        parallel_workers=script_arguments.parallel_workers,
        output_knowledge_path=script_arguments.output_knowledge_file
    )