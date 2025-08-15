import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import re
import json
import os
from utils.api_clients import openai_chat
from tqdm import tqdm
from multiprocessing import Pool
from utils.prompt_processing import get_prompt_template
from config import SUBTOPIC_SELECTION_MODEL
import csv
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Subtopic mapping with configurable parameters')
    parser.add_argument('--book_path', nargs="+",
                       help='Path to the book JSON file, can be multiple. E.g. resources/bio_book_raw.json')
    parser.add_argument('--embedding_model', type=str, default='Qwen/Qwen3-Embedding-4B',
                       help='Embedding model to use')
    parser.add_argument('--curriculum_csv', type=str,
                       help='Path to the curriculum mapping CSV file. E.g. resources/curriculum.csv')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top similar chunks to retrieve')
    parser.add_argument('--output_mapping_csv', type=str, required=True,
                       help='Path to save the new CSV with textbook_subtopics column. E.g. resources/curriculum_mapping.csv. The mapping might not be perfect, so expert review is recommended.')
    return parser.parse_args()

def subtopic_mapping(subtopic, content):
    prompt_template = get_prompt_template("prompts/chunk_retrieval.yaml", "subtopic_mapping")
    prompt = prompt_template.format(subtopic=subtopic, content=content)
    response = openai_chat(prompt, model=SUBTOPIC_SELECTION_MODEL, max_tokens=10000).strip()
    pattern = re.compile(r'^\d+\.\s+(.*)', re.MULTILINE)
    subtopics = pattern.findall(response)
    return [s.strip() for s in subtopics if s.strip()]

def main():
    args = parse_args()

    model = SentenceTransformer(args.embedding_model)
    
    books = []
    for path in args.book_path:
        books += json.load(open(path))
    
    chunks = []
    chunk_to_subtopic = {}

    for topic in books:
        for subtopic in topic['subtopics']:
            chunk_sum = subtopic['name'] + ' -- ' + subtopic['summary']
            chunks.append(chunk_sum)
            chunk_to_subtopic[chunk_sum] = subtopic['name']
    
    chunks = list(dict.fromkeys(chunks))
    chunk_embeddings = model.encode(chunks, batch_size=16, max_length=40000, show_progress_bar=True)

    with open(args.curriculum_csv, 'r', encoding='utf-8') as f_in, \
         open(args.output_mapping_csv, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['textbook_subtopics']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader):
            topic = row['cur_topic']
            subtopic = row['cur_subtopic']
            subtopic_query = row['cur_outcome']

            # embed the subtopic query
            subtopic_query_embedding = model.encode([subtopic_query], prompt_name="query")[0]

            # select top k most similar chunks
            similarity = model.similarity(subtopic_query_embedding, chunk_embeddings)[0]
            top_k_indices = list(reversed(similarity.argsort()[-args.top_k:]))
            top_k_chunks = [chunks[idx] for idx in top_k_indices]

            content_to_prompt = ""

            print("Outcome:", subtopic_query)
            for chunk in top_k_chunks:
                chunk_title = chunk_to_subtopic[chunk]
                content_to_prompt += "Section title: " + chunk_title + "\n"
                content_to_prompt += "Section summary: " + chunk + "\n\n---\n\n"
            
            textbook_subtopics = subtopic_mapping(subtopic_query, content_to_prompt)
            print("Mapped sutopics:", textbook_subtopics)
            print('\n------------------------------\n')
            
            out_row = dict(row)
            out_row['textbook_subtopics'] = ' | '.join(textbook_subtopics)
            writer.writerow(out_row)

    print(f"\nProcessing complete!")
    print(f"New CSV saved to: {args.output_mapping_csv}")

if __name__ == "__main__":
    main()