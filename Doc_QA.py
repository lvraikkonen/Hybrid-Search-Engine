import cohere

from utils.db_client import get_es_client, get_milvus_client
from utils.get_text_embedding import get_text_embedding_ada_v2 as get_text_embedding
from utils.chat_wrapper import chat_completion

from utils.logger import logger
from config.config_parser import MILVUS_SIZE, ES_SIZE, MILVUS_THRESHOLD, COHERE_API_KEY, RERANK_TOP_N


class DocQA:
    def __init__(self, model_name: str = "gpt-3.5-turbo-1106"):
        self.model_name = model_name

    # get search result from vector-store Milvus
    def get_milvus_search_result(self, query):
        milvus_client = get_milvus_client("docs_qa")
        # embedding query
        vector_to_search = [get_text_embedding(query)]
        search_params = {
            "metric_type": "IP",  # or COSINE
            "params": {"nprobe": 10},
        }
        result = milvus_client.search(
            vector_to_search,
            "embeddings",  # queried field
            search_params,
            limit=MILVUS_SIZE,
            output_fields=["text", "source"]
        )

        # filter by similarity score
        return [(_.entity.get('text'), _.entity.get('source'))
                for _, dist in zip(result[0], result[0].distances) if dist > MILVUS_THRESHOLD]

    # get search result form Elasticsearch BM25 keyword-search
    def get_elasticsearch_search_result(self, query):
        es_client = get_es_client()
        result = []
        dsl = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": ES_SIZE
        }
        search_result = es_client.search(index="docs_qa", body=dsl)
        if search_result['hits']['hits']:
            result = [(_['_source']['content'], _['_source']['source']) for _ in search_result['hits']['hits']]
        return result

    def get_context(self, query):
        contents = []
        milvus_search_result = self.get_milvus_search_result(query)
        es_search_result = self.get_elasticsearch_search_result(query)

        # remove duplicate
        for content_source_tuple in milvus_search_result + es_search_result:
            content, source = content_source_tuple
            if [content, source] not in contents:
                contents.append([content, source])
        return contents

    def rerank(self, query):
        before_rerank_contents = self.get_context(query)

        # rerank using CoHere rerank API
        cohere_client = cohere.Client(COHERE_API_KEY)
        docs, sources = [_[0] for _ in before_rerank_contents], [_[1] for _ in before_rerank_contents]

        results = cohere_client.rerank(
            model="rerank-multilingual-v2.0",
            query=query,
            documents=docs,
            top_n=RERANK_TOP_N
        )
        after_rerank_contents = []
        for hit in results:
            after_rerank_contents.append([hit.document['text'], sources[hit.index]])
            logger.info(f"Score: {hit.relevance_score}, query: {query}, text: {hit.document['text']}")

        return after_rerank_contents

    def get_qa_prompt(self, query):
        # 建立prompt
        prefix = "<文本片段>:\n\n"
        suffix = f"\n<问题>: {query}\n<回答>: "
        prompt = []
        contexts = []
        sources = []
        for i, text_source_tuple in enumerate(self.rerank(query)):
            text, source = text_source_tuple
            prompt.append(f"{i + 1}: {text}\n")
            contexts.append(f"<{i + 1}>: {text}")
            sources.append(f"<{i + 1}>: {source}")
        qa_chain_prompt = prefix + ''.join(prompt) + suffix
        contexts, sources = "\n\n".join(contexts), "\n\n".join(sources)
        logger.info(qa_chain_prompt)
        return qa_chain_prompt, contexts, sources

    def answer(self, query):
        message, contexts, sources = self.get_qa_prompt(query=query)
        result = chat_completion(message, self.model_name)
        return result, contexts, sources
    

if __name__ == '__main__':
    test_model_name = 'gpt-3.5-turbo-1106'
    question = "请介绍一下OpenAI发布的GPT商店"
    result= DocQA(model_name=test_model_name).answer(query=question)
    print(result)