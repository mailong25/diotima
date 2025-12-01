import yaml
from langchain.prompts import PromptTemplate
from utils.api_clients import llm_generate_response
from config import MAX_RETRIES
import random

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

def eval_candidate(task_prompt, task_answer, eval_model):

    def parse_evaluation(response):
        for line in response.splitlines():
            if line.lower().startswith("final decision:"):
                decision = line.split(":", 1)[1].strip().strip('.')
                if decision in ["Accept", "Reject"]:
                    return decision

        print('\n\n-----------------', response)
        return None
    
    return run_prompt(
        template_path="prompts/evaluation.yaml",
        template_key="binary_classification",
        template_vars={"task_prompt": task_prompt, "task_answer": task_answer},
        model=eval_model,
        base_temperature=0.0,
        temperature_strategy="increasing",
        parse_response_func=parse_evaluation,
    )

def run_synthesis(
    template_path: str,
    template_key: str,
    template_vars: dict,
    models: list,
    base_temperature: float = 0.0,
    temperature_strategy: str = "fixed",
    parse_response_func=None,
    trace: list = None,
):
    """
    General-purpose majority synthesis with evaluation & regeneration.

    Args:
        template_path (str): Path to the YAML prompt file.
        template_key (str): Key inside the YAML file.
        template_vars (dict): Variables to format into the template.
        models (list[str]): List of models in "<provider>/<model_name>" format.
            - If length == 1: falls back to run_prompt (no voting).
            - If odd length >= 3: uses majority synthesis.
        base_temperature (float): Temperature for candidate generation.
        temperature_strategy (str): "fixed" or "increasing".
        parse_response_func (callable): Optional response parser.
        trace (list): Optional list to append trace records for debugging/logging.

    Returns:
        Parsed accepted response, or None if all retries fail.
    """
    if not models or not isinstance(models, list):
        raise ValueError("models must be a non-empty list")
    
    # Case 1: single model → fallback to run_prompt
    if len(models) == 1:
        return run_prompt(
            template_path=template_path,
            template_key=template_key,
            template_vars=template_vars,
            model=models[0],
            base_temperature=base_temperature,
            temperature_strategy=temperature_strategy,
            parse_response_func=parse_response_func,
        )

    # Case 2: odd number of models → majority synthesis
    if len(models) % 2 == 0:
        raise ValueError("Majority synthesis requires an odd number of models (3, 5, 7, ...)")

    prompt_template = get_prompt_template(template_path, template_key)
    task_prompt = prompt_template.format(**template_vars)

    for attempt in range(1, MAX_RETRIES + 1):
        selected_model = random.choice(models)

        candidate_response = run_prompt(
            template_path=template_path,
            template_key=template_key,
            template_vars=template_vars,
            model=selected_model,
            base_temperature=base_temperature,
            temperature_strategy=temperature_strategy,
            parse_response_func=parse_response_func,
        )

        # Evaluation phase
        votes = []
        for eval_model in models:
            decision = eval_candidate(task_prompt, candidate_response, eval_model)
            votes.append((eval_model, decision))

        eval_votes = [d for _, d in votes]
        accept_count = eval_votes.count("Accept")
        majority_needed = len(models) // 2 + 1   # strict majority (e.g. 2 of 3, 3 of 5)
        accepted = accept_count >= majority_needed

        # Record traceability
        if trace is not None:
            trace.append({
                "attempt": attempt,
                "selected_model": selected_model,
                "candidate": candidate_response,
                "votes": votes,
                "accepted": accepted,
            })

        if accepted:
            return candidate_response

    return None