from typing import List
from operator import itemgetter

from llama_index.schema import TextNode, NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping
from custom_retrievers.bm25_retriever import CustomBM25Retriever
from custom_retrievers.vector_store_retriever import VectorSearchRetriever


class EnsembleRetriever(BaseRetriever):
    def __init__(self, top_k, faiss_index, weights):
        self.weights = weights
        self.c: int = 60
        self.faiss_index = faiss_index
        self.top_k = top_k
        self.vector_store_retriever = VectorSearchRetriever(top_k=self.top_k, faiss_index=self.faiss_index)
        self.bm25_retriever = CustomBM25Retriever(top_k=self.top_k)

    def _retrieve(self, query_bundle: QueryType) -> List[NodeWithScore]:
        bm25_search_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_search_nodes = self.vector_store_retriever.retrieve(query_bundle)

        bm25_docs = [node.text for node in bm25_search_nodes]
        vector_docs = [node.text for node in vector_search_nodes]
        doc_lists = [bm25_docs, vector_docs]

        # remove duplicates
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        # Calculate RRF score for each document
        rrf_score_dict = {doc: 0.0 for doc in all_documents}

        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dict[doc] += rrf_score

        # sort documents by their RRF score in desc
        sorted_documents = sorted(rrf_score_dict.items(), key=itemgetter(1), reverse=True)
        result = []
        for sorted_doc in sorted_documents[:self.top_k]:
            text, score = sorted_doc
            node_with_score = NodeWithScore(node=TextNode(text=text,
                                                          id_=text_node_id_mapping[text]),
                                            score=score)
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from faiss import IndexFlatIP

    faiss_index = IndexFlatIP(1536)
    query = "中国队亚洲杯成绩如何"
    ensemble_retriever = EnsembleRetriever(top_k=3, faiss_index=faiss_index, weights=[0.5, 0.5])
    t_result = ensemble_retriever.retrieve(str_or_query_bundle=query)
    print(t_result)
    faiss_index.reset()
