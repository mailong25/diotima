from utils.prompt_processing import run_prompt
from config import RUBRIC_CLASSIFICATION_MODEL

def classify_student_answer(question, rubric, student_answer):
    rubric_text = "\n".join([f"  - {level}: {description}" for level, description in rubric.items()])

    def parse_classification(response):
        for line in response.splitlines():
            if line.lower().startswith("rubric level:"):
                rubric_level = line.split(":", 1)[1].strip().strip('.')
                if rubric_level in rubric:
                    return rubric_level
        return None
    
    return run_prompt(
        template_path="prompts/rubric_classification.yaml",
        template_key="rubric_classification",
        template_vars={"question": question, "rubric": rubric_text, "student_answer": student_answer},
        model=RUBRIC_CLASSIFICATION_MODEL,
        base_temperature=0.0,
        temperature_strategy="increasing",
        parse_response_func=parse_classification,
    )
