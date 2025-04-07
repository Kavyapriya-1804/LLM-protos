from langchain_groq import ChatGroq


def create_chat_groq_model(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
):
    """
    Creates and returns a configured instance of the ChatGroq model.

    Args:
        model -> str: The model to use (default: "mixtral-8x7b-32768").
        temperature -> float: Sampling temperature for randomness (default: 0).
        max_tokens -> int or None: Maximum number of tokens to generate (default: None).
        timeout -> int or None: Timeout for requests in seconds (default: None).
        max_retries -> int: Number of retries on request failures (default: 2).

    Returns:
        ChatGroq: Configured ChatGroq model instance
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries
    )
