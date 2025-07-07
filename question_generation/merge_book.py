import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_book.py file1.json,file2.json,... [output_file.json]")
        return
    
    raw_args = sys.argv[1]
    files = raw_args.split(',')

    output_file = sys.argv[2] if len(sys.argv) > 2 else 'merged_book.json'

    merge_book = []

    for f in files:
        print("Reading book:", f)
        book_name = f.split('/')[-1].replace('.json', '')
        with open(f, 'r') as file:
            book = json.load(file)

        merge_book.append({
            'name': book_name,
            'content': book
        })
    
    try:
        with open(output_file, 'w') as file:
            json.dump(merge_book, file, indent=4)
        print(f"Merged book saved to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


if __name__ == "__main__":
    main()
