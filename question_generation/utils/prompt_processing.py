import yaml
from langchain.prompts import PromptTemplate
def get_prompt_template(template_path, prompt_name):
    with open(template_path, "r") as f:
        templates = yaml.safe_load(f)
    
    template_data = templates[prompt_name]
    return PromptTemplate(
        input_variables=template_data["input_variables"],
        template=template_data["template"]
    )