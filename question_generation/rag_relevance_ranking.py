#!/usr/bin/env python3
"""
RAG Knowledge Mapping Script
Maps curriculum topics to relevant knowledge chunks using semantic similarity and LLM relevance checking.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.prompt_processing import get_prompt_template

from config import RELEVANCE_CHECK_MODEL, QUERY_EXPANSION_MODEL
from utils.api_clients import openai_chat

class CurriculumKnowledgeMapper:
    """Handles curriculum-to-knowledge mapping operations using RAG."""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.knowledge_chunks = []
        self.knowledge_embeddings = None
        
        # Priority order for relevance scores (lower number = higher priority)
        self.relevance_priority = {
            'Very high relevance': 1,
            'High relevance': 2, 
            'Medium relevance': 3,
        }
    
    def _initialize_embedding_model(self) -> None:
        """Load and configure the sentence transformer model."""
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda", dtype=torch.float16) 
    
    def _load_knowledge_base(self) -> None:
        """Load and preprocess knowledge chunks from source files."""
        print("Loading knowledge base from files...")
        raw_chunks = []
        
        for file_path in self.config.source_files:
            if not Path(file_path).exists():
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read().split('\n\n')
                raw_chunks.extend(file_content)
        
        # Filter out short chunks and remove duplicates
        MIN_WORDS = 10
        self.knowledge_chunks = [chunk for chunk in raw_chunks if len(chunk.split()) >= MIN_WORDS]
        self.knowledge_chunks = list(dict.fromkeys(self.knowledge_chunks))
        
        print(f"Loaded {len(self.knowledge_chunks)} unique knowledge chunks")
    
    def _create_knowledge_embeddings(self) -> None:
        """Generate embeddings for all knowledge chunks."""
        print("Creating embeddings for knowledge chunks...")
        self.knowledge_embeddings = self.embedding_model.encode(
            self.knowledge_chunks, 
            batch_size=16, 
            max_length=20000,
            show_progress_bar=True
        )
    
    def generate_related_queries(self, original_query: str) -> List[str]:
        """Generate additional related queries using LLM to improve coverage."""
        prompt_template = get_prompt_template('prompts/chunk_retrieval.yaml', 'query_expansion')
        formatted_prompt = prompt_template.format(query=original_query)
        llm_response = openai_chat(formatted_prompt, model=QUERY_EXPANSION_MODEL, max_tokens=10000)
        
        # Extract numbered queries from LLM response
        query_pattern = re.compile(r'^\d+\.\s+(.*)', re.MULTILINE)
        extracted_queries = query_pattern.findall(llm_response)
        return [query.strip() for query in extracted_queries if query.strip()]
    
    def evaluate_chunk_relevance(self, evaluation_request: Dict) -> Optional[str]:
        """Evaluate if a knowledge chunk is relevant to a query using LLM."""
        prompt_template = get_prompt_template('prompts/chunk_retrieval.yaml', 'relevant_chunk')
        formatted_prompt = prompt_template.format(
            chunk=evaluation_request['chunk'], 
            query=evaluation_request['query']
        )
        llm_response = openai_chat(formatted_prompt, model=RELEVANCE_CHECK_MODEL, max_tokens=10000)
        
        # Extract conclusion from LLM response
        conclusion_match = re.search(r'Conclusion:\s*(.*)', llm_response)
        return conclusion_match.group(1).strip() if conclusion_match else None
    
    def find_similar_chunks(self, query: str) -> List[Dict]:
        """Find most semantically similar chunks for a given query."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], prompt_name="query")[0]
        
        # Calculate similarities and get top candidates
        similarities = self.embedding_model.similarity(query_embedding, self.knowledge_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:self.config.top_k]
        candidate_chunks = [self.knowledge_chunks[idx] for idx in top_indices]
        
        # Evaluate relevance using parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluation_requests = [
                {'chunk': chunk, 'query': query} 
                for chunk in candidate_chunks
            ]
            relevance_results = list(executor.map(self.evaluate_chunk_relevance, evaluation_requests))
        
        # Compile results with relevance scores
        relevant_chunks = []
        for chunk, relevance_score in zip(candidate_chunks, relevance_results):
            if relevance_score in self.relevance_priority:
                relevant_chunks.append({
                    'content': chunk,
                    'relevance_level': relevance_score,
                    'source_query': query,
                    'chunk_index': self.knowledge_chunks.index(chunk),
                })
        
        return relevant_chunks
    
    def process_curriculum_item(self, curriculum_row: Dict) -> Tuple[str, str, List[Dict]]:
        """Process a single curriculum topic and find relevant knowledge chunks."""
        topic_name = curriculum_row['cur_topic']
        subtopic_name = curriculum_row['cur_subtopic']
        learning_outcome = curriculum_row['cur_outcome']
        
        # Generate comprehensive query set
        related_queries = self.generate_related_queries(learning_outcome)
        all_queries = [learning_outcome] + related_queries[:9]  # Limit to 10 total queries
        
        print(f"Processing: {topic_name} -> {subtopic_name}")
        
        # Collect unique chunks with best relevance scores
        best_chunks = {}
        
        for current_query in all_queries:
            similar_chunks = self.find_similar_chunks(current_query)
            
            for chunk_data in similar_chunks:
                chunk_content = chunk_data['content']
                
                # Keep only the highest relevance version of each chunk
                if (chunk_content not in best_chunks or 
                    self.relevance_priority[chunk_data['relevance_level']] < 
                    self.relevance_priority[best_chunks[chunk_content]['relevance_level']]):
                    best_chunks[chunk_content] = chunk_data
        
        # Sort by relevance priority (best first)
        prioritized_chunks = list(best_chunks.values())
        prioritized_chunks.sort(key=lambda x: self.relevance_priority[x['relevance_level']])
        
        return topic_name, subtopic_name, prioritized_chunks
    
    def map_entire_curriculum(self) -> Dict:
        """Process the complete curriculum CSV and map to knowledge chunks."""
        curriculum_mapping = defaultdict(dict)
        
        if not Path(self.config.curriculum_csv).exists():
            raise FileNotFoundError(f"Curriculum CSV not found: {self.config.curriculum_csv}")
        
        with open(self.config.curriculum_csv, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for curriculum_row in tqdm(csv_reader, desc="Mapping curriculum to knowledge"):
                topic, subtopic, relevant_chunks = self.process_curriculum_item(curriculum_row)
                curriculum_mapping[topic][subtopic] = relevant_chunks
                
                # Save progress incrementally
                self._write_json_file(dict(curriculum_mapping), self.config.detailed_output_path)
        
        return dict(curriculum_mapping)
    
    def create_condensed_knowledge_base(self, full_mapping: Dict) -> Dict:
        """Create condensed knowledge base containing only high-relevance content."""
        condensed_knowledge = {}
        
        for topic_name in full_mapping:
            condensed_knowledge[topic_name] = {}
            
            for subtopic_name in full_mapping[topic_name]:
                chunk_list = full_mapping[topic_name][subtopic_name]
                
                # Filter for high-relevance chunks only
                high_relevance_chunks = [
                    chunk for chunk in chunk_list 
                    if 'high relevance' in chunk.get('relevance_level', '').lower()
                ]
                
                # Sort by original index and combine content
                high_relevance_chunks.sort(key=lambda x: x['chunk_index'])
                combined_content = '\n\n'.join(
                    chunk['content'] for chunk in high_relevance_chunks
                ).strip()
                
                # Only include if content meets minimum length requirement
                MIN_COMBINED_WORDS = 10
                if len(combined_content.split()) >= MIN_COMBINED_WORDS:
                    condensed_knowledge[topic_name][subtopic_name] = combined_content
        
        return condensed_knowledge
    
    def _write_json_file(self, data_to_save: Dict, output_path: str) -> None:
        """Save data to JSON file with proper error handling."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data_to_save, json_file, indent=2, ensure_ascii=False)
        except Exception as write_error:
            print(f"Error saving to {output_path}: {write_error}")
    
    def execute_mapping_pipeline(self) -> None:
        """Execute the complete curriculum-to-knowledge mapping pipeline."""
        try:
            # Initialize components
            self._initialize_embedding_model()
            self._load_knowledge_base()
            self._create_knowledge_embeddings()
            
            # Execute mapping process
            full_curriculum_mapping = self.map_entire_curriculum()
            print(f"Successfully mapped {len(full_curriculum_mapping)} curriculum topics")
            
            # Generate condensed output
            condensed_output = self.create_condensed_knowledge_base(full_curriculum_mapping)
            self._write_json_file(condensed_output, self.config.condensed_output_path)
            
            print(f"Pipeline completed. Results saved to:")
            print(f"  - Detailed mapping: {self.config.detailed_output_path}")
            print(f"  - Condensed knowledge base: {self.config.condensed_output_path}")
        
        except Exception as pipeline_error:
            print(f"Pipeline execution failed: {pipeline_error}")
            raise

def parse_command_line_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Map curriculum topics to relevant knowledge chunks using RAG',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--embedding-model', 
        type=str, 
        default="Qwen/Qwen3-Embedding-4B",
        help='Name of the SentenceTransformer embedding model to use'
    )
    
    parser.add_argument(
        '--source-files', 
        type=str, 
        nargs='+', 
        help='List of text files containing knowledge content to map from. E.g. resources/bio_book_knowledge.txt'
    )
    
    parser.add_argument(
        '--curriculum-csv', 
        type=str, 
        default='resources/subtopic_learning_outcome.csv',
        help='CSV file containing curriculum topics and learning outcomes. E.g. resources/curriculum.csv'
    )
    
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help='Number of most similar chunks to evaluate for relevance per query'
    )
    
    parser.add_argument(
        '--detailed-output-path', 
        type=str, 
        help='Output path for detailed curriculum mapping with all relevance levels. resources/detailed_rag_mapping.json'
    )
    
    parser.add_argument(
        '--condensed-output-path', 
        type=str, 
        default='resources/condensed_knowledge_base.json',
        help='Output path for condensed knowledge base with high-relevance content only. E.g. resources/structured_book_rag.json'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the curriculum knowledge mapping script."""
    configuration = parse_command_line_arguments()
    knowledge_mapper = CurriculumKnowledgeMapper(configuration)
    knowledge_mapper.execute_mapping_pipeline()


if __name__ == "__main__":
    main()