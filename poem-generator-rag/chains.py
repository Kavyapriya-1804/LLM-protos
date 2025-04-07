from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb


#### GENERATION ####
def generate_poem_chain(topic):
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


#### RETRIEVAL and GENERATION ####

def generate_poem_rag_chain(topic, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        topic - topic for retrieval
        vectorstore ->  Instance of vector store 

    Returns:
        rag_chain -> rag chain
    """
    # Prompt
    prompt = prompts.poem_generator_rag_prompt

    # LLM
    llm = models.create_chat_groq_model()

    # Post-processing
    def format_docs(docs):
        print(vectordb.retrieve_from_chroma(topic))
        return "\n\n".join(doc.page_content for doc in docs)
    
    # vectordb.retrieve_from_chroma(topic)

    # Chain
    rag_chain = (
        {"context": format_docs(topic), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
