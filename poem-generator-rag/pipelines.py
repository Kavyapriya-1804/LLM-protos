
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

import models
import prompts


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


def generate_poem_rag():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    #Initializing the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Intializing the vector store
    vector_store = Chroma(
        collection_name="poem_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    assert len(example_messages) == 1
    print(example_messages[0].content)

generate_poem_rag()