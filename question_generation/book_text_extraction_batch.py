#!/usr/bin/env python3
import argparse
import json
import time
from openai import OpenAI
from utils.pdf_utils import pdf_to_text
from utils.text_processing import text_to_chunks
from utils.prompt_processing import get_prompt_template
from config import KNOWLEDGE_EXTRACTION_MODEL, OPENAI_API_KEY
import uuid

MAX_WORDS_PER_CHUNK = 2000

def build_batch_file(text_chunks, prompt_template_path, output_jsonl_path):
    """Builds the .jsonl file for Batch API input."""
    prompt_template = get_prompt_template(prompt_template_path, 'knowledge_extraction_and_paragraphing')

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(text_chunks):
            formatted_prompt = prompt_template.format(text=chunk)
            request = {
                "custom_id": f"chunk-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": KNOWLEDGE_EXTRACTION_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that extracts structured knowledge from text."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    "max_tokens": 16384
                }
            }
            f.write(json.dumps(request) + "\n")

    print(f"✅ Batch input file created: {output_jsonl_path} ({len(text_chunks)} requests)")


def submit_batch_job(client, batch_input_file_id):
    """Submits the batch job to OpenAI."""
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "knowledge extraction batch job"}
    )
    print(f"📦 Batch submitted: {batch.id} (status: {batch.status})")
    return batch.id


def wait_for_batch_completion(client, batch_id, poll_interval=60):
    """Polls the batch status until completion."""
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"⏳ Batch status: {batch.status}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch
        time.sleep(poll_interval)


def extract_knowledge_from_pdf_batch(source_pdf_path, start_page, end_page, output_knowledge_path):
    """Pipeline using OpenAI Batch API instead of parallel workers."""
    print("🗂️ Extracting text from PDF pages...")
    raw_pdf_text = pdf_to_text(source_pdf_path, start_page, end_page)

    print("✂️ Splitting text into manageable chunks...")
    text_chunks = text_to_chunks(raw_pdf_text, chunk_max_words=MAX_WORDS_PER_CHUNK, splitter=".\n")
    print(len(text_chunks))
    
    print(f"🔹 Created {len(text_chunks)} chunks. Building batch input file...")
    batch_input_path = "batchinput_" + uuid.uuid4().hex + ".jsonl"
    build_batch_file(text_chunks, 'prompts/chunk_retrieval.yaml', batch_input_path)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("📤 Uploading batch input file to OpenAI...")
    batch_input_file = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")

    print("🚀 Creating batch job...")
    batch_id = submit_batch_job(client, batch_input_file.id)

    print("⏳ Waiting for batch completion (can take up to 24h)...")
    completed_batch = wait_for_batch_completion(client, batch_id)

    if completed_batch.status != "completed":
        print(f"❌ Batch did not complete successfully. Status: {completed_batch.status}")
        return

    print("📥 Downloading results...")
    output_file_id = completed_batch.output_file_id
    file_response = client.files.content(output_file_id)

    # Parse JSONL responses
    extracted_knowledge_chunks = []
    for line in file_response.text.strip().split("\n"):
        data = json.loads(line)
        response = data.get("response", {})
        if response and "body" in response:
            message = response["body"]["choices"][0]["message"]["content"]
            extracted_knowledge_chunks.append(message.strip())

    print(f"🧩 Retrieved {len(extracted_knowledge_chunks)} responses from batch.")

    consolidated_knowledge = "\n\n".join(extracted_knowledge_chunks)

    with open(output_knowledge_path, "w", encoding="utf-8") as out_file:
        out_file.write(consolidated_knowledge)

    print(f"✅ Knowledge extraction complete. Output saved to: {output_knowledge_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured knowledge from PDFs using OpenAI Batch API")
    parser.add_argument("--source-pdf", required=True)
    parser.add_argument("--start-page", type=int, required=True)
    parser.add_argument("--end-page", type=int, required=True)
    parser.add_argument("--output-knowledge-file", required=True)
    args = parser.parse_args()

    extract_knowledge_from_pdf_batch(
        source_pdf_path=args.source_pdf,
        start_page=args.start_page,
        end_page=args.end_page,
        output_knowledge_path=args.output_knowledge_file
    )
