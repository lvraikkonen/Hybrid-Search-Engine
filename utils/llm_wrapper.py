# from __future__ import annotations

import json
from openai import OpenAI

from config.config_parser import SYSTEM_ROLE, OPENAI_API_KEY
from utils.logger import logger


class AnLLM:
    """
    This class provides a consistent API for the different LLM services.
    Intended usage:

        from utils.llm_wrapper import OpenAILLM, AitomaticLLM, AzureLLM

        llm1 = OpenAILLM.get_default()
        llm1.call(messages=[{"role": "user", "content": "Say this is a test"}], stream=True)
        llm1.create_embeddings()

    etc.
    """

    def __init__(
        self,
        model: str = None,
        # api_base: str = None,
        api_key: str = None,
        **additional_kwargs,
    ):
        self.model = model
        # self.api_base = api_base
        self.api_key = api_key
        self._client = None
        self._additional_kwargs = additional_kwargs

    @property
    def client(self):
        pass

    def call(self, is_chat: bool = True, **kwargs):
        if is_chat:
            result = self.client.chat.completions.create(
                model=self.model, **kwargs, **self._additional_kwargs
            )
        else:
            result = self.client.completions.create(
                model=self.model, **kwargs, **self._additional_kwargs
            )
        return result

    def create_embeddings(self, input_text):
        return self.client.embeddings.create(model=self.model, input=input_text)

    def get_response(
        self, prompt: str, messages: list = None, role: str = "user", **kwargs
    ) -> str:
        messages = messages if messages else []
        messages.append({"role": role, "content": prompt})
        return self.call(messages=messages, **kwargs).choices[0].message.content

    def parse_output(self, output: str) -> dict:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            print("Failed to decode the response as JSON.")
            return {}


class OpenAILLM(AnLLM):
    """
    This class represents the OpenAI-hosted LLMs
    """

    def __init__(
        self,
        model: str = None,
        # api_base: str = None,
        api_key: str = None,
        **additional_kwargs,
    ):
        if model is None:
            model = "gpt-3.5-turbo-1106"
        # if api_base is None:
        #     api_base = Config.OPENAI_API_URL
        if api_key is None:
            api_key = OPENAI_API_KEY
        super().__init__(
            model=model, api_key=api_key, **additional_kwargs
        )

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @classmethod
    def get_default(cls):
        return cls.get_gpt_35_turbo()

    @classmethod
    def get_gpt_35_turbo_1106(cls):
        return cls(model="gpt-3.5-turbo-1106")

    @classmethod
    def get_gpt_35_turbo_0613(cls):
        return cls(model="gpt-3.5-turbo")

    @classmethod
    def get_gpt_35_turbo(cls):
        return cls(model="gpt-3.5-turbo-0613")

    @classmethod
    def get_gpt_4(cls):
        return cls(model="gpt-4")

    @classmethod
    def get_gpt_4_1106_preview(cls):
        return cls(model="gpt-4-1106-preview")
    
    @classmethod
    def get_gpt_4_0125_preview(cls):
        return cls(model="gpt-4-0125-preview")

    @classmethod
    def get_embedding_model_v2(cls):
        return cls(model="text-embedding-ada-002")
    
    @classmethod
    def get_embedding_model_v3_small(cls):
        return cls(model="text-embedding-3-small")


if __name__ == '__main__':
    messages = [{"role": "system", "content": SYSTEM_ROLE}, {"role": "user", "content": "你好~"}]
    # llm1 = OpenAILLM.get_gpt_4()
    # chat_response = llm1.get_response(prompt="你好~")  # llm1.call(messages=messages, stream=False)
    # print(chat_response)
    embed_model = OpenAILLM.get_embedding_model_v2()
    embeddings = embed_model.create_embeddings(input_text="你好")
    print(embeddings)
