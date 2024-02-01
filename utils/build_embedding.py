import os
import json
import numpy as np
from tqdm import tqdm
from utils.get_text_embedding import (
    get_text_embedding_ada_v2,
    get_text_embedding_v3,
    get_bge_embedding
)


def build_with_context(embedding_model: str, context_type: str):
    with open("../data/doc_qa_dataset.json", "r", encoding="utf-8") as f:
        content = json.loads(f.read())
    queries = list(content[context_type].values())
    query_num = len(queries)
    if embedding_model == 'text-embedding-ada-002':
        embedding_fn, dimensions = get_text_embedding_ada_v2, 1536
    elif embedding_model == 'text-embedding-3-small':
        embedding_fn, dimensions = get_text_embedding_v3, 1536
    elif embedding_model == 'text-embedding-3-large':
        embedding_fn, dimensions = get_text_embedding_v3, 3072
    elif embedding_model == 'local_bge_zh-v1.5':
        embedding_fn, dimensions = get_bge_embedding, 1024

    embedding_data = np.empty(shape=[query_num, dimensions])
    for i in tqdm(range(query_num), desc="generate embedding"):
        embedding_data[i] = embedding_fn(queries[i])
    np.save(f"../data/{context_type}_{embedding_model}.npy", embedding_data)


class EmbeddingCache:
    """
    Generate embedding cache
    """
    def __init__(self, embedding_provider="OpenAI"):
        self.embedding_provider = embedding_provider  # default using text-embedding-ada-002 from OpenAI

    @staticmethod
    def build():
        build_with_context("text-embedding-ada-002", "queries")
        build_with_context("text-embedding-ada-002", "corpus")

    @staticmethod
    def load(embedding_model: str = 'text-embedding-ada-002', query_write=False):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        queries_embedding_data = np.load(os.path.join(current_dir, f"data/queries_{embedding_model}.npy"))
        corpus_embedding_data = np.load(os.path.join(current_dir, f"data/corpus_{embedding_model}.npy"))
        query_embedding_dict = {}
        with open(os.path.join(current_dir, "data/doc_qa_dataset.json"), "r", encoding="utf-8") as f:
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
    EmbeddingCache.load("text-embedding-ada-002")
