import re

def numbered_to_list(text):
    items = re.findall(r'\d+\.\s+(.*)', text)
    return [t.strip() for t in items]
