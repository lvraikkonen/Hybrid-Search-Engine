from utils.logger import logger
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus import utility
from elasticsearch import Elasticsearch

# 连接Elasticsearch
es_client = Elasticsearch("http://localhost:9200")


def initialize_es(index_name):
    # 创建新的ES index
    mapping = {
        'properties': {
            'source': {
                'type': 'text',
                'fields': {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            'cont_id': {
                'type': 'integer'
            },
            'content': {
                'type': 'text',
                'analyzer': 'ik_smart',
                'search_analyzer': 'ik_smart'
            },
            'file_type': {
                'type': 'keyword'
            },
            "insert_time": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            }
        }
    }

    es_client.indices.create(index=index_name, ignore=400)
    result = es_client.indices.put_mapping(index=index_name, body=mapping)

    logger.info(f"Elasticsearch index {index_name} created Successfully. With es return {result}")


def initialize_vector_store_milvus(collection_name):
    connections.connect("default", host="localhost", port="19530")
    # Create a collection
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100, description="source file metadata"),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000, description="raw text"),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536,
                    description="store embedding of text array")
    ]
    schema = CollectionSchema(fields, "vector db for docs qa")
    docs_milvus = Collection(collection_name, schema)

    # 在embeddings字段创建索引
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }
    docs_milvus.create_index("embeddings", index)

    logger.info(f"Milvus Index {collection_name} created successfully.")


def get_es_client():
    return es_client


def get_milvus_client(collection_name):
    connections.connect("default", host="localhost", port="19530")
    return Collection(collection_name)


if __name__ == '__main__':
    # index_name = 'docs_qa'
    # initialize_es(index_name)

    vector_collection = "docs_qa"
    connections.connect("default", host="localhost", port="19530")
    # # # drop existing collection
    # # utility.drop_collection(vector_collection)
    #
    # initialize_vector_store_milvus(vector_collection)
    # collection = Collection(vector_collection)
    # print(collection.schema)

    collection = Collection(vector_collection)
    print(f"There are {collection.num_entities} entities in Milvus.")
