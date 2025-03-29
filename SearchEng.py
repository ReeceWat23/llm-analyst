import openai

import embeddings
from pymilvus import MilvusClient


client = MilvusClient(
    uri="https://in03-5d7a6e0d5ff229c.api.gcp-us-west1.zillizcloud.com",  # Cluster endpoint obtained from the console
    token="db3f305a3164011c78212dffbb378983bd31bc209e0722d35623447320de893808c6873c5fc33a484ebd8f526e13c01aba81d033"
)

openai.api_key = "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"


def search_db(text,collection):
    # create an embedding
    vect = embeddings.create_embedding(text)
    # create an embedding
    results = client.search(collection_name=collection,data=[vect],limit=3,output_fields=["Content","source"])
    return results[0]


def ask_llm(text,knowledge_base):
    # define the instructions for our llm
    SystemContent = """You are a helpful research assistant for investment research. You will help answer users questions based on your knowledge base, provide
     detailed responses, provide analysis on content in your knowledge base & use your knowledge of financial terms, and methods to help. If you do not know the answer, say you do not know the answer. You should only answer questions with data in your knowledge base.
     - text will be in the 'content' section & sources in the 'source' section of the database'"""

    user_content = f"""
    Knowledge base:
    {knowledge_base}
    ----------
    Query:
    {text}
    ----------
    Answer:
    """

    system_message = {"role": "system", "content": SystemContent}
    user_message = {"role": "user", "content": user_content}

    chat_gpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system_message,user_message],)
    return chat_gpt_response.choices[0].message.content