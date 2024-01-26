################# Rerank module ###############

import os
import time
from pydantic import BaseModel
from operator import itemgetter
from random import randint
import cohere
from typing import List, Tuple
# from retry import retry

from FlagEmbedding import FlagReranker

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

class QuerySuite(BaseModel):
    query: str
    passages: List[str]
    top_k: int = 3


# Using Cohere Rerank service
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

def get_cohere_rerank_result(query: str, docs: List[str], top_n) -> List[Tuple]:
    time.sleep(randint(1, 10))

    results = cohere_client.rerank(
        model="rerank-multilingual-v2.0",
        query=query,
        documents=docs,
        top_n=top_n
    )

    # return top_n nodes text with relevence score
    return [(hit.document['text'], hit.relevance_score) for hit in results]


def get_bge_rerank_result(query: str, docs: List[str], top_n) -> List[Tuple]:
    query_suite = QuerySuite(
        query=query,
        passages=docs
    )

    bge_reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    scores = bge_reranker.compute_score([[query_suite.query, passage] for passage in query_suite.passages])

    if isinstance(scores, List):
        similiarity_dict = {passage: scores[i] for i, passage in enumerate(query_suite.passages)}
    else:
        similarity_dict = {passage: scores for i, passage in enumerate(query_suite.passages)}
    
    sorted_similarity_dict = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True) # sorted by score
    result = {}
    for j in range(query_suite.top_k):
        result[sorted_similarity_dict[j][0]] = sorted_similarity_dict[j][1]
    
    # return top_n nodes text with relevence score
    return [(passage, score) for passage, score in result.items()]