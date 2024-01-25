from typing import List

import numpy as np
from llama_index.schema import TextNode
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping

from utils.db_client import get_milvus_client
from utils.get_text_embedding import get_text_embedding
from config.config_parser import (
    MILVUS_SIZE, MILVUS_THRESHOLD, RERANK_TOP_N
)


class VectorSearchRetriever(BaseRetriever):
    def __init__(self, collection_name, top_k, query_rewrite=False) -> None:
        super().__init__()
        self.collection_name = collection_name
        self.top_k = top_k
        self.milvus_client = get_milvus_client(self.collection_name)

    def _retrieve(self, query: QueryBundle) -> List[NodeWithScore]:
        if isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            query_str = query

        result = []
        vector_to_search = [get_text_embedding(query_str)]
        search_params = {
            "metric_type": "IP",  # or COSINE
            "params": {"nprobe": 10},
        }
        # vector search with output raw text and source-metadata
        search_result = self.milvus_client.search(
            vector_to_search,
            "embeddings",  # queried field
            search_params,
            limit=MILVUS_SIZE,
            output_fields=["text", "source"]
        )
        
        # filter by similarity score
        # filtered_result = [(_.entity.get('text'), _.entity.get('source'))
        #         for _, dist in zip(search_result[0], search_result[0].distances) if dist > MILVUS_THRESHOLD]

        for _, dist in zip(search_result[0], search_result[0].distances):
            text = _.entity.get('text')
            score = dist
            source = _.entity.get('source')
            node_with_score = NodeWithScore(
                node=TextNode(text=text),
                id_=text_node_id_mapping[text],
                score=score
            )
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from pprint import pprint
    vector_search_retriever = VectorSearchRetriever(collection_name="docs_qa", top_k=3)
    query = "中国队亚洲杯成绩如何"
    t_result = vector_search_retriever.retrieve(str_or_query_bundle=query)
    pprint(t_result)