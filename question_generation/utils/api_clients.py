import openai
from mistralai import Mistral
import time
from config import OPENAI_API_KEY, MISTRAL_API_KEY

def mistralai_chat(prompt, model="mistral-large-latest", temperature=0, max_tokens=10000):
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def openai_chat(prompt, model="gpt-4o", temperature=0, max_tokens=10000):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    if "gpt-5" in model:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
    return response.choices[0].message.content
