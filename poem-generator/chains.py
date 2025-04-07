from dotenv import load_dotenv
import models
import prompts

load_dotenv()


def generate_poem(topic):
    """
    Generate Poem using basic prompt LLM chain

    Args:
        topic - topic for the poem

    Returns:
        response.content -> str
    """
        
    llm = models.create_chat_groq_model()

    prompt_template = prompts.poem_generator_prompt()
    # prompt_template = prompts.poem_generator_prompt_from_hub()

    chain = prompt_template | llm
    response = chain.invoke({
        "topic" : topic
    })
    return response.content
