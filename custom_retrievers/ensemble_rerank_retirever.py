from typing import List

from llama_index.schema import TextNode, NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryBundle

from preprocess.get_text_id_mapping import text_node_id_mapping
from custom_retrievers.bm25_retriever import CustomBM25Retriever
from custom_retrievers.vector_store_retriever import VectorSearchRetriever

from utils.rerank import get_cohere_rerank_result


class EnsembleRerankRetriever(BaseRetriever):
    def __int__(self, top_k, faiss_index):
        super.__init__()
        self.faiss_index = faiss_index
        self.top_k = top_k
        self.vector_store_retriever = VectorSearchRetriever(top_k=self.top_k, faiss_index=self.faiss_index)
        self.bm25_retriever = CustomBM25Retriever(top_k=self.top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print(query_bundle.query_str)
        bm25_search_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_search_nodes = self.vector_store_retriever.retrieve(query_bundle)

        bm25_docs = [node.text for node in bm25_search_nodes]
        vector_docs = [node.text for node in vector_search_nodes]
        # remove duplicate document
        all_documents = set()
        for doc_list in [bm25_docs, vector_docs]:
            for doc in doc_list:
                all_documents.add(doc)
        doc_lists = list(all_documents)
        # rerank using query and docs
        rerank_doc_lists = get_cohere_rerank_result(query_bundle.query_str, doc_lists, top_n=self.top_k)
        result = []
        for sorted_doc in rerank_doc_lists:
            text, score = sorted_doc
            node_with_score = NodeWithScore(node=TextNode(text=text,
                                                          id_=text_node_id_mapping[text]),
                                            score=score)
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from faiss import IndexFlatIP

    faiss_index = IndexFlatIP(1536)
    ensemble_rerank_retriever = EnsembleRerankRetriever(top_k=2, faiss_index=faiss_index)
    t_result = ensemble_rerank_retriever.retrieve(str_or_query_bundle="中国队亚洲杯成绩如何")
    print(t_result)
    faiss_index.reset()
