import os
import time
import json
import numpy as np
from retry import retry
from tqdm import tqdm
from utils.get_text_embedding import get_text_embedding


class EmbeddingCache():
    """
    Generate embedding cache
    """
    def __init__(self, embedding_provider = "OpenAI"):
        self.embedding_provider = embedding_provider # default using text-embedding-ada-002 from OpenAI

    def build_with_context(self, context_type: str):
        with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content[context_type].values())
        query_num = len(queries)
        embedding_data = np.empty(shape=[query_num, 1536]) # 1536 is for text-embedding-ada-002
        for i in tqdm(range(query_num), desc="generate embedding"):
            embedding_data[i] = self.get_bge_embedding(queries[i])
        np.save(f"../data/{context_type}_bge_base_ft_embedding.npy", embedding_data)

    def build(self):
        self.build_with_context("queries")
        self.build_with_context("corpus")

    @staticmethod
    def load(query_write=False):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        queries_embedding_data = np.load(os.path.join(current_dir, "data/queries_bge_base_ft_embedding.npy"))
        corpus_embedding_data = np.load(os.path.join(current_dir, "data/corpus_bge_base_ft_embedding.npy"))
        query_embedding_dict = {}
        with open(os.path.join(current_dir, "data/doc_qa_test.json"), "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content["queries"].values())
        corpus = list(content["corpus"].values())
        for i in range(len(queries)):
            query_embedding_dict[queries[i]] = queries_embedding_data[i].tolist()
        if query_write:
            rewrite_queries_embedding_data = np.load(os.path.join(current_dir, "data/query_rewrite_openai_embedding.npy"))
            with open("../data/query_rewrite.json", "r", encoding="utf-8") as f:
                rewrite_content = json.loads(f.read())

            rewrite_queries_list = []
            for original_query, rewrite_queries in rewrite_content.items():
                rewrite_queries_list.extend(rewrite_queries)
            for i in range(len(rewrite_queries_list)):
                query_embedding_dict[rewrite_queries_list[i]] = rewrite_queries_embedding_data[i].tolist()
        return query_embedding_dict, corpus_embedding_data, corpus


if __name__ == '__main__':
    EmbeddingCache().build()