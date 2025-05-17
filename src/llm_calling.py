from typing import Any, List, Dict, Optional

def run_llm_call(
    openai_client: Any,
    model: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    """
    Sends a basic chat message to a language model (no tool calling).

    Args:
        openai_client (Any): Initialized OpenAI client (e.g., from OpenAI or OpenRouter).
        model (str): Model name to use (e.g., 'openai/gpt-3.5-turbo').
        messages (List[Dict[str, str]]): List of chat messages in OpenAI format.

    Returns:
        Optional[str]: The assistant's response content, or None if no response is returned.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )

    message = response.choices[0].message
    return message.content
