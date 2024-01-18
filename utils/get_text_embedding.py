import json
import requests

from config.config_parser import (EMBEDDING_API,
                                  OPENAI_API_KEY)


# default using OpenAI text embedding
def get_text_embedding(req_text, model_name = "text-embedding-ada-002"):
    headers = {'Content-Type': 'application/json'}
    embedding_api = EMBEDDING_API
    
    headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    
    payload = json.dumps({"model": model_name, "input": req_text})
    new_req = requests.request("POST", embedding_api, headers=headers, data=payload)
    return new_req.json()['data'][0]['embedding']

if __name__=='__main__':
    result = get_text_embedding("I live in Beijing.")
    print(result)