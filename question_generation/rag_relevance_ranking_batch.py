#!/usr/bin/env python3
"""
RAG Question Relevance Mapping (Automatic Batch API version)
------------------------------------------------------------
This script:
  • Loads one or more knowledge text files
  • Embeds all chunks with a SentenceTransformer model
  • Expands each question into sub-queries
  • Finds the top-50 most similar chunks across all sub-queries
  • Creates an OpenAI Batch API job to evaluate relevance
  • Waits for completion, downloads results, and outputs a JSON file

Example:
python rag_question_relevance_batch_autorun.py \
  --source-files resources/book_knowledge.txt resources/extra_notes.txt \
  --questions-csv resources/questions.csv \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --top-k 10 \
  --output-json resources/final_question_relevance.json
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from utils.prompt_processing import get_prompt_template
from utils.api_clients import openai_chat
from config import RELEVANCE_CHECK_MODEL, QUERY_EXPANSION_MODEL


class RAGQuestionRelevanceBatchAuto:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI()
        self.embedding_model = None
        self.knowledge_chunks = []
        self.knowledge_embeddings = None

    # -------------------------------------------------------------------------
    # === INITIALIZATION ===
    # -------------------------------------------------------------------------
    def _initialize_embedding_model(self):
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda", dtype=torch.float16)

    def _load_knowledge_base(self):
        print("Loading knowledge base files...")
        raw_chunks = []
        for path in self.config.source_files:
            p = Path(path)
            if not p.exists():
                print(f"⚠️ Skipping missing file {path}")
                continue
            with open(p, "r", encoding="utf-8") as f:
                raw_chunks.extend(f.read().split("\n\n"))
        self.knowledge_chunks = list(
            dict.fromkeys([c for c in raw_chunks if len(c.split()) >= 10])
        )
        print(f"Loaded {len(self.knowledge_chunks)} unique chunks.")

    def _create_embeddings(self):
        print("Creating embeddings for knowledge base...")
        self.knowledge_embeddings = self.embedding_model.encode(
            self.knowledge_chunks, batch_size=16, show_progress_bar=True
        )

    # -------------------------------------------------------------------------
    # === QUERY EXPANSION (SYNC) ===
    # -------------------------------------------------------------------------
    def generate_related_queries(self, question: str) -> List[str]:
        """Use LLM synchronously to generate related queries."""
        prompt_template = get_prompt_template('prompts/chunk_retrieval.yaml', 'query_expansion')
        formatted_prompt = prompt_template.format(query=question)
        llm_response = openai_chat(formatted_prompt, model=QUERY_EXPANSION_MODEL, max_tokens=2000)

        query_pattern = re.compile(r'^\d+\.\s+(.*)', re.MULTILINE)
        related = query_pattern.findall(llm_response)
        return [q.strip() for q in related if q.strip()]

    # -------------------------------------------------------------------------
    # === BATCH CREATION ===
    # -------------------------------------------------------------------------
    def build_relevance_requests(self, questions: List[str]) -> List[Dict]:
        """Generate all relevance evaluation prompts for the Batch API."""
        requests = []
        req_count = 0

        for qi, question in enumerate(tqdm(questions, desc="Building relevance requests")):
            related = self.generate_related_queries(question)
            queries = [question] + related[:9]

            # --- Step 1: Gather all (chunk_idx, similarity_score) from all sub-queries ---
            all_candidates = []
            for q in queries:
                q_emb = self.embedding_model.encode([q])[0]
                sims = self.embedding_model.similarity(q_emb, self.knowledge_embeddings)[0]
                top_indices = sims.argsort(descending=True)[: self.config.top_k]
                for idx in top_indices.tolist():
                    all_candidates.append((idx, float(sims[idx])))

            # --- Step 2: Deduplicate, keeping highest similarity per chunk ---
            chunk_best_sim = {}
            for idx, sim_score in all_candidates:
                if idx not in chunk_best_sim or sim_score > chunk_best_sim[idx]:
                    chunk_best_sim[idx] = sim_score

            # --- Step 3: Sort by similarity, keep top 50 ---
            top_candidates = sorted(chunk_best_sim.items(), key=lambda x: x[1], reverse=True)[:50]

            # --- Step 4: Create relevance-check requests ---
            for rank, (idx, sim_score) in enumerate(top_candidates):
                chunk = self.knowledge_chunks[idx]
                prompt_template = get_prompt_template('prompts/chunk_retrieval.yaml', 'relevant_chunk')
                prompt = prompt_template.format(chunk=chunk, query=question)

                req_count += 1
                requests.append({
                    "custom_id": f"relevance_q{qi}_c{idx}_r{req_count}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.config.relevance_check_model,
                        "messages": [
                            {"role": "system", "content": "You are an expert evaluator."},
                            {"role": "user", "content": prompt}
                        ],
                    }
                })

        print(f"Prepared {len(requests)} relevance evaluation requests.")
        return requests

    def create_and_run_batch(self, requests: List[Dict]) -> str:
        """Write JSONL, upload to OpenAI, and create batch."""
        jsonl_path = "relevance_batch_input.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for obj in requests:
                f.write(json.dumps(obj) + "\n")
        print(f"📁 Wrote batch input file: {jsonl_path}")

        file_obj = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "RAG relevance evaluation batch"},
        )

        print(f"✅ Batch created: {batch.id}")
        return batch.id

    def wait_for_batch_completion(self, batch_id: str, poll_interval: int = 120):
        """Poll batch status until completion or failure."""
        print(f"⏳ Waiting for batch {batch_id} to complete...")
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            print(f"  Status: {status}")
            if status == "completed":
                print("✅ Batch completed!")
                return batch
            elif status in ["failed", "cancelled", "expired"]:
                raise RuntimeError(f"❌ Batch ended with status: {status}")
            time.sleep(poll_interval)

    def download_batch_results(self, batch):
        """Download the output file from a completed batch."""
        output_file_id = batch.output_file_id
        if not output_file_id:
            raise RuntimeError("No output_file_id found in batch.")
        print(f"📥 Downloading results from {output_file_id} ...")
        file_response = self.client.files.content(output_file_id)
        output_path = "relevance_batch_output.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(file_response.text)
        print(f"✅ Saved batch output to {output_path}")
        return output_path

    # -------------------------------------------------------------------------
    # === POST-PROCESSING ===
    # -------------------------------------------------------------------------
    @staticmethod
    def parse_batch_line(line: str):
        """Extract custom_id and conclusion label."""
        try:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            message = obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            match = re.search(r"Conclusion:\s*(.*)", message)
            return cid, match.group(1).strip() if match else None
        except Exception:
            return None, None

    def process_batch_output(self, output_file: str, questions_file: str, output_json: str):
        """Process relevance_batch_output.jsonl into final structured JSON."""
        print(f"🔍 Reading batch output from {output_file}")
        results = {}
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                cid, relevance = self.parse_batch_line(line)
                if cid and relevance:
                    results[cid] = relevance
        print(f"Parsed {len(results)} relevance labels.")

        # Reload questions
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

        # Map back to questions/chunks
        question_map = {i: q for i, q in enumerate(questions)}
        mapping = {}
        for cid, label in results.items():
            m = re.search(r"q(\d+)_c(\d+)", cid)
            if not m:
                continue
            qi, ci = int(m.group(1)), int(m.group(2))
            question = question_map.get(qi)
            if not question or ci >= len(self.knowledge_chunks):
                continue
            chunk = self.knowledge_chunks[ci]
            mapping.setdefault(question, []).append({"content": chunk, "relevance_level": label})

        # Filter and sort
        priority = {"Very high relevance": 1, "High relevance": 2, "Medium relevance": 3}
        final_data = []
        for q, chunks in mapping.items():
            filtered = [c for c in chunks if "high" in c["relevance_level"].lower()]
            filtered.sort(key=lambda x: priority.get(x["relevance_level"], 99))
            final_data.append({"question": q, "relevant_paragraphs": filtered})

        # Save final JSON
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved final relevance mapping to {output_json}")
        print(f"Contains {len(final_data)} questions.")

    # -------------------------------------------------------------------------
    # === MAIN DRIVER ===
    # -------------------------------------------------------------------------
    def run(self):
        self._initialize_embedding_model()
        self._load_knowledge_base()
        self._create_embeddings()

        with open(self.config.questions_csv, "r", encoding="utf-8") as f:
            questions = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

        requests = self.build_relevance_requests(questions)
        batch_id = self.create_and_run_batch(requests)
        batch = self.wait_for_batch_completion(batch_id)
        output_file = self.download_batch_results(batch)
        self.process_batch_output(output_file, self.config.questions_csv, self.config.output_json)


# -------------------------------------------------------------------------
# === CLI ===
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Automatic RAG Batch API Question Relevance Mapping")
    parser.add_argument("--source-files", type=str, nargs="+", required=True,
                        help="Knowledge text files used for chunk retrieval.")
    parser.add_argument("--questions-csv", type=str, required=True,
                        help="CSV file with one question per line.")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--relevance-check-model", type=str, default=RELEVANCE_CHECK_MODEL)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top chunks per sub-query before merging (default: 10)")
    parser.add_argument("--output-json", type=str, default="resources/final_question_relevance.json",
                        help="Where to save the final processed JSON mapping.")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    job = RAGQuestionRelevanceBatchAuto(cfg)
    job.run()
