from typing import List

import numpy as np
from llama_index.schema import TextNode
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping
from utils.build_embedding import EmbeddingCache

from utils.db_client import get_milvus_client
from utils.get_text_embedding import get_text_embedding_ada_v2
from config.config_parser import (
    MILVUS_SIZE, MILVUS_THRESHOLD, RERANK_TOP_N
)


class VectorSearchRetriever(BaseRetriever):
    def __init__(self, top_k, faiss_index, query_rewrite=False) -> None:
        super().__init__()
        self.faiss_index = faiss_index
        self.top_k = top_k
        # get query, corpus embedding cache for simulating query VectorStore
        self.queries_embedding_dict, self.corpus_embedding, self.corpus = EmbeddingCache().load(
            query_write=query_rewrite)
        self.faiss_index.add(self.corpus_embedding)

    def _retrieve(self, query_bundle: QueryType) -> List[NodeWithScore]:
        result = []
        # query embedding data
        if query_bundle.query_str in self.queries_embedding_dict:
            query_embedding = self.queries_embedding_dict[query_bundle.query_str]
        else:
            query_embedding = get_text_embedding_ada_v2(req_text=query_bundle.query_str)

        distances, doc_indices = self.faiss_index.search(np.array([query_embedding]), self.top_k)

        for i, sent_index in enumerate(doc_indices.tolist()[0]):
            text = self.corpus[sent_index]
            node_with_score = NodeWithScore(node=TextNode(text=text, id_=text_node_id_mapping[text]),
                                            score=distances.tolist()[0][i])
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from pprint import pprint
    from faiss import IndexFlatIP
    faiss_index = IndexFlatIP(1536)  # ada-002
    vector_search_retriever = VectorSearchRetriever(top_k=3, faiss_index=faiss_index)
    query = "中国队亚洲杯成绩如何"
    t_result = vector_search_retriever.retrieve(str_or_query_bundle=query)
    pprint(t_result)
