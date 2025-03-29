import openai
import tiktoken

openai.api_key = "key here"
# make it so a user can enter in their own api key


def create_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings


# free
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


