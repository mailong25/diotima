import re

def numbered_to_list(text):
    items = re.findall(r'\d+\.\s+(.*)', text)
    return [t.strip() for t in items]

# Split text into chunks of max words while respecting paragraph boundaries
def text_to_chunks(text, chunk_max_words=2000, splitter=".\n"):
    paragraphs = text.split(splitter)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_word_count = len(paragraph_words)
        
        # If adding this paragraph would exceed the limit, start a new chunk
        if current_word_count + paragraph_word_count > chunk_max_words and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(splitter.join(current_chunk))
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(splitter.join(current_chunk))
    
    return chunks

def get_last_index(text, sub_string):
    sub_pattern = r"\s+".join(sub_string.lower().split())
    matches = list(re.finditer(sub_pattern, text, flags=re.IGNORECASE))
    if matches:
        last_index = matches[-1].start()
        return last_index
    else:
        return -1