import json
import yaml
from langchain.prompts import PromptTemplate

from utils.api_clients import openai_chat
from utils.text_processing import numbered_to_list

def get_relevant_text(book, topic, subtopic=None):
    with open("prompts/rag_prompt.yaml", "r") as f:
        templates = yaml.safe_load(f)

    template_data = templates['unit_extraction']
    prompt_template = PromptTemplate(
        template=template_data["template"],
        input_variables=template_data["input_variables"]
    )

    units = book
    if subtopic is not None:
        # Flatten all subtopics into a single list
        units = [sub for unit in book for sub in unit['subtopics']]
    else:
        # Combine all subtopic texts for each main topic
        for unit in units:
            unit['text'] = '\n\n'.join(sub['text'] for sub in unit['subtopics'])

    tab_content = '\n'.join([unit['name'] for unit in units])
    prompt = prompt_template.format(
        topic=subtopic if subtopic else topic,
        tab_content=tab_content
    )

    resp = openai_chat(prompt)
    selected_units = numbered_to_list(resp)

    relevant_text = []
    for unit in units:
        if unit['name'] in selected_units:
            relevant_text.append('\n-------' + unit['name'] + '-------\n')
            relevant_text.append(unit['text'])

    return '\n'.join(relevant_text), selected_units
