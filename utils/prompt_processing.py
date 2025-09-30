import yaml
from langchain.prompts import PromptTemplate
from utils.api_clients import llm_generate_response
from config import MAX_RETRIES

def get_prompt_template(template_path, prompt_name):
    with open(template_path, "r") as f:
        templates = yaml.safe_load(f)
    
    template_data = templates[prompt_name]
    return PromptTemplate(
        input_variables=template_data["input_variables"],
        template=template_data["template"]
    )

def run_prompt(
    template_path: str,
    template_key: str,
    template_vars: dict,
    model: str,
    base_temperature: float = 0.0,
    temperature_strategy: str = "fixed",
    parse_response_func=None,
):
    """
    Run a prompt with retries and optional custom parsing.

    Args:
        template_path (str): Path to the YAML prompt file.
        template_key (str): Key inside the YAML file.
        template_vars (dict): Variables to format into the template.
        model (str): Model name to use. Format: provider/model_name
        base_temperature (float): Default temperature.
        temperature_strategy (str): "fixed" or "increasing": control randomness of the regenerated response
        parse_response (callable): Optional function(response: str) -> any.

    Returns:
        Parsed response (str or custom type) or None if all retries fail.
    """
    prompt_template = get_prompt_template(template_path, template_key)
    prompt = prompt_template.format(**template_vars)

    for attempt in range(1, MAX_RETRIES + 1):
        if temperature_strategy == "increasing":
            temperature = base_temperature if attempt == 1 else min(base_temperature + 0.1 * attempt, 1.0)
        else:
            temperature = base_temperature

        response = llm_generate_response(prompt, model=model, temperature=temperature).strip()
        if response:
            return parse_response_func(response) if parse_response_func else response

    return None
