from utils.prompt_processing import run_prompt
from config import FEEDBACK_GENERATION_MODEL

def generate_feedback(question, rubric, rubric_category, reference_answer, student_answer, student_reading_level):
    rubric_text = "\n".join([f"  - {level}: {description}" for level, description in rubric.items()])
    
    return run_prompt(
        template_path="prompts/feedback_generation.yaml",
        template_key="feedback_generation",
        template_vars={
            "question": question,
            "rubric": rubric_text,
            "rubric_category": rubric_category,
            "reference_answer": reference_answer,
            "student_answer": student_answer,
            "student_reading_level": student_reading_level,
        },
        model=FEEDBACK_GENERATION_MODEL,
    )