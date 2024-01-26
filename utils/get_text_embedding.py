import json
import requests
from retry import retry
from openai import OpenAI

from config.config_parser import (EMBEDDING_API,
                                  OPENAI_API_KEY)


# default using OpenAI text embedding
@retry(exceptions=Exception, tries=3, max_delay=60)
def get_text_embedding_ada_v2(req_text, model_name="text-embedding-ada-002"):
    model = model_name
    input = req_text

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=model,
        input=input,
        encoding_format="float"
    )
    return response.json()['data'][0]['embedding']


# using OpenAI text embedding V3
@retry(exceptions=Exception, tries=3, max_delay=60)
def get_text_embedding_v3_small(req_text, model_name="text-embedding-3-small"):
    model = model_name
    input = req_text

    if model in ['text-embedding-3-large', 'text-embedding-3-small']:  #  3-large with size: 3072
        dimensions = 1536

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=model,
        input=input,
        encoding_format="float",
        dimensions=dimensions
    )
    return response.json()['data'][0]['embedding']


@retry(exceptions=Exception, tries=3, max_delay=60)
def get_bge_embedding(req_text: str):
    # localhost embedding service
    url = "http://localhost:50072/embedding"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"text": req_text})
    new_req = requests.request("POST", url, headers=headers, data=payload)
    return new_req.json()['embedding']


if __name__ == '__main__':
    result = get_text_embedding_v3_small("I live in Beijing.")
    print(result)
