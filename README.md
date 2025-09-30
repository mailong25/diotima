## 📦 Installation

pip install -r requirements.txt

## Example Use

```
from rubric_classification import classify_student_answer
from feedback_generation import generate_feedback

question = "Describe the role of mitochondria in cellular respiration."

rubric = {
    "Emerging": "Shows minimal understanding; major misconceptions or irrelevant response.",
    "Developing": "Demonstrates basic understanding; contains some errors or incomplete explanation.",
    "Proficient": "Shows good understanding; mostly accurate with minor omissions or inaccuracies.",
    "Advanced": "Demonstrates excellent understanding with a complete and accurate explanation."
}

student_answer = "Mitochondria are parts of something that I just forgot."
student_reading_level = "16-18 years old"

reference_answer = """Mitochondria are the “powerhouses” of the cell. 
They break down glucose and oxygen to produce ATP through cellular respiration, 
mainly in the Krebs cycle and electron transport chain. 
This ATP is the energy the cell uses to do work."""

# Step 1: Classify student response
rubric_category = classify_student_answer(question, rubric, student_answer)
print("Rubric Category:", rubric_category)

# Step 2: Generate personalized feedback
feedback = generate_feedback(
    question,
    rubric,
    rubric_category,
    reference_answer,
    student_answer,
    student_reading_level
)
print("Feedback:", feedback)
```