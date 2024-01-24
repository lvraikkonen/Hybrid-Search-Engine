import json

import requests

from config.config_parser import SYSTEM_ROLE, CHAT_COMPLETION_API, OPENAI_API_KEY

from utils.logger import logger


def chat_completion(message, model_name):
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_ROLE
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    )
    chat_completion_api = CHAT_COMPLETION_API
    headers = {"Content-Type": 'application/json', "Authorization": f"Bearer {OPENAI_API_KEY}"}

    response = requests.request(
        "POST",
        chat_completion_api,
        headers=headers,
        data=payload
    )
    logger.info(f"Model_Name: {model_name}, response: {response.text}")
    return response.json()['choices'][0]['message']['content']


if __name__ == '__main__':
    completion = chat_completion("你好", "gpt-3.5-turbo")
    print(completion)
