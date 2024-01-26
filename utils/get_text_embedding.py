import json
from utils.logger import logger

import requests
from retry import retry
from openai import OpenAI

from config.config_parser import OPENAI_API_KEY


# default using OpenAI text embedding
@retry(exceptions=Exception, tries=3, max_delay=60)
def get_text_embedding_ada_v2(req_text, model_name="text-embedding-ada-002"):
    model = model_name
    input_text = req_text.replace("\n", " ")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=model,
        input=input_text,
        encoding_format="float"
    )
    embeddings = response.data[0].embedding
    return embeddings


# using OpenAI text embedding V3
@retry(exceptions=Exception, tries=3, max_delay=60)
def get_text_embedding_v3(req_text, model_name="text-embedding-3-small"):
    model = model_name
    input_text = req_text.replace("\n", " ")

    if model in ['text-embedding-3-large', 'text-embedding-3-small']:  # 3-large with size: 3072
        dimensions = 1536

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=model,
        input=input_text,
        encoding_format="float",
        # dimensions=dimensions
    )
    embeddings = response.data[0].embedding
    logger.info(f"Embedding input with {embeddings}")
    return embeddings


@retry(exceptions=Exception, tries=3, max_delay=60)
def get_bge_embedding(req_text: str):
    # localhost embedding service
    url = "http://localhost:50072/embedding"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"text": req_text})
    new_req = requests.request("POST", url, headers=headers, data=payload)
    return new_req.json()['embedding']


if __name__ == '__main__':
    result = get_text_embedding_v3(req_text="I live in Beijing.", model_name='text-embedding-3-small')
    print(f"embedding text with dimension {len(result)}")
