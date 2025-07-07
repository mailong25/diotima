import json
import csv
import argparse
from collections import defaultdict

def build_subtopic_mapping(books):
    """Create mapping from subtopic names to their text content."""
    subtopic_to_text = {}
    
    for book in books:
        book_name = book['name']
        for topic in book['content']:
            for subtopic in topic['subtopics']:
                full_name = f"{book_name}: {subtopic['name']}"
                subtopic_to_text[full_name] = subtopic['text']
    
    return subtopic_to_text

def build_curriculum_mapping(csv_filepath, subtopic_to_text):
    mapped_curriculum = defaultdict(dict)

    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            topic = row['cur_topic']
            subtopic = row['cur_subtopic']

            subtopic_texts = []
            for i in range(1, 11):
                key = f'textbook_subtopic_{i}'
                subtopic_key = row.get(key, '')
                if subtopic_key and subtopic_key in subtopic_to_text:
                    header = f"\n-------{subtopic_key.split(': ')[-1]}-------\n"
                    subtopic_texts.extend([header, subtopic_to_text[subtopic_key]])

            if subtopic_texts:
                mapped_curriculum[topic][subtopic] = '\n'.join(subtopic_texts)
            else:
                print(f"Warning: No text found for subtopic '{subtopic}'")

    return dict(mapped_curriculum)

def main(book_path, mapping_path, output_path):
    with open(book_path, 'r') as f:
        books = json.load(f)

    subtopic_to_text = build_subtopic_mapping(books)
    mapped_curriculum = build_curriculum_mapping(mapping_path, subtopic_to_text)

    with open(output_path, 'w') as f:
        json.dump(mapped_curriculum, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map curriculum subtopics to textbook text.")
    parser.add_argument('--book_path', type=str, required=True, help="Path to book JSON file")
    parser.add_argument('--mapping_path', type=str, required=True, help="Path to mapping CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output JSON file")

    args = parser.parse_args()
    main(args.book_path, args.mapping_path, args.output_path)
