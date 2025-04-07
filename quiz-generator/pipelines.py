from dotenv import load_dotenv
import models
import prompts

load_dotenv()



def generate_quiz(topic, number_of_questions, difficulty):
    """
    Generate Quiz using basic prompt LLM chain

    Args:
        topic - topic for the quiz
        number_of_questions - number of questions in the quiz
        difficulty - difficulty level

    Returns:
        response.content -> str
    """

    llm = models.create_chat_groq_model()

    prompt_template = prompts.quiz_generator_prompt()
    # prompt_template = prompts.quiz_generator_prompt_from_hub()

    chain = prompt_template | llm

    response = chain.invoke({
        "topic" : topic,
        "number_of_questions" : number_of_questions,
        "difficulty" : difficulty
    })
    return response.content
