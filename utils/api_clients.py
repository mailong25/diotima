import os
import openai
from mistralai import Mistral
from config import MISTRAL_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

def mistralai_chat(
    prompt,
    model="mistral-medium-latest",
    temperature=0,
    max_tokens=10000,
    api_key=None,
):
    api_key = api_key or MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key not provided and not found in environment variables.")

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def openai_chat(
    prompt,
    model="gpt-4o",
    temperature=0,
    max_tokens=10000,
    api_key=None,
):
    api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and not found in environment variables.")

    client = openai.OpenAI(api_key=api_key)

    kwargs = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if "gpt-5" not in model:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    
    return response.choices[0].message.content

def gemini_chat(
    prompt,
    model = "gemini-2.5-flash",
    temperature = 0.0,
    max_tokens = 10000,
    api_key = None,
) -> str:
    
    api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required or must be provided.")

    client = genai.Client(api_key=api_key)

    thinking_budget = 100
    config = GenerateContentConfig(
        temperature=temperature,
        maxOutputTokens=max_tokens + thinking_budget,
        thinking_config=ThinkingConfig(thinking_budget=thinking_budget),
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    return response.text.strip()

# -------- Factory Dispatcher -------- #

def llm_generate_response(
    prompt,
    model="openai/gpt-4.1",
    temperature=0,
    max_tokens=10000,
    api_key=None,
):
    """
    Unified dispatcher for different LLM providers (OpenAI, Mistral, Gemini).

    Args:
        prompt (str): The input prompt for the model.
        model (str): Provider/model identifier. Example:
            - "openai/gpt-4.1"
            - "mistral/mistral-medium-latest"
            - "google/gemini-2.5-flash"
        temperature (float): Sampling temperature.
        max_tokens (int): Max tokens to generate.
        api_key (str, optional): API key override.

    Returns:
        str: Generated text response.
    """
    if not isinstance(model, str) or "/" not in model:
        raise ValueError("Model must be specified in the format '<provider>/<model_name>'.")

    provider, model_name = model.split("/", 1)

    if provider == "openai":
        return openai_chat(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    elif provider == "mistral":
        return mistralai_chat(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    elif provider == "google":
        return gemini_chat(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Must be one of: openai, mistral, google.")