# Using GPT-4 for generating qa dataset
import os
import json

import pandas as pd
from llama_index.llms import OpenAI
from llama_index.schema import TextNode
from llama_index.evaluation import generate_question_context_pairs
import random
from dotenv import load_dotenv
random.seed(42)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=openai_api_key, model="gpt-4", max_retries=5)


def convert_unicode_to_utf8(json_file):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 转换所有字符串值
    def convert_item(item):
        if isinstance(item, dict):
            return {key: convert_item(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [convert_item(element) for element in item]
        elif isinstance(item, str):
            return item.encode('utf-8').decode('utf-8')
        else:
            return item

    converted_data = convert_item(data)

    # 将转换后的数据写回文件
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(converted_data, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Prompt to generate questions
    qa_generate_prompt_tmpl = """\
    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    Given the context information and not prior knowledge.
    generate only questions based on the below query.

    You are a news reporter. Your task is to set {num_questions_per_chunk} questions for the upcoming Chinese quiz.
    Questions throughout the test should be diverse. Questions should not contain options or start with Q1/Q2.
    Questions must be written in Chinese. The expression must be concise and clear.
    It should not exceed 15 Chinese characters. Words such as "这", "那", "根据", "依据" and other punctuation marks
    should not be used. Abbreviations may be used for titles and professional terms.
    """

    nodes = []
    data_df = pd.read_csv("data/docs_qa.csv", encoding="utf-8")
    for i, row in data_df.iterrows():
        if len(row["content"]) > 80:
            node = TextNode(text=row["content"])
            node.id_ = f"node_{i + 1}"
            nodes.append(node)

    doc_qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
    )

    doc_qa_dataset.save_json("data/doc_qa_dataset.json")

    convert_unicode_to_utf8("data/doc_qa_dataset.json")
