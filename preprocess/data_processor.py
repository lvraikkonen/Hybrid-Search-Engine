import datetime
from elasticsearch import helpers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter

from utils.db_client import get_milvus_client, get_es_client
from utils.get_text_embedding import get_text_embedding_ada_v2 as get_text_embedding
from utils.logger import logger
from file_parser import FileParser

from retry import retry


class DataProcessor(object):
    """
    数据处理, 分别加载进入ES和向量数据库
    """
    def __init__(self, file_path, file_content=""):
        self.file_path = file_path
        self.file_content = file_content

    def text_loader(self):
        logger.info(f'loading file: {self.file_path}')
        documents, file_type = FileParser(self.file_path, self.file_content).parse()
        return documents, file_type

    @staticmethod
    def text_spliter(documents):
        # 将文档拆分成块
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=256)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=60)
    def text_embedding(texts):
        # calc embeddings 
        # return [ids, source_metadata, origin_content, embedding] to store
        _ids = []
        sources = []
        contents = []
        embeddings = []
        for i, text in enumerate(texts):
            source = text.metadata['source']
            content = text.page_content
            content = content.replace('\n', '')
            embedding = get_text_embedding(content)
            _ids.append(i + 1)
            sources.append(source)
            contents.append(content)
            embeddings.append(embedding)
            logger.info(f'source: {source}, got text {i} embedding...')
        datas = [_ids, sources, contents, embeddings]
        return datas

    @staticmethod
    def es_data_insert(datas, file_type):
        es_client = get_es_client()
        if datas:
            action = ({
                "_index": "docs_qa",
                "_source": {
                    "source": datas[1][i],  # file source metadata
                    "cont_id": datas[0][i],  # content id
                    "content": datas[2][i],  # raw content data
                    "file_type": file_type,
                    "insert_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            } for i in range(len(datas[0])))
            # use the bulk helper which can index Elasticsearch documents from iterators or generators.
            helpers.bulk(es_client, action)
            logger.info("Docs data have inserted into elasticsearch")
        else:
            logger.info("no insert data!")

    @staticmethod
    def milvus_data_insert(datas):
        milvus_client = get_milvus_client("docs_qa")
        insert_result = milvus_client.insert(datas)
        milvus_client.flush()
        # 将collection加载至内存
        milvus_client.load()
        logger.info(f"insert data to milvus, {insert_result}")

    def data_insert(self):
        documents, file_type = self.text_loader()
        texts = self.text_spliter(documents)
        datas = self.text_embedding(texts)
        self.es_data_insert(datas, file_type)
        self.milvus_data_insert(datas)


if __name__ == '__main__':
    DataProcessor(file_path='../data/files/asian_cup_2023.txt').data_insert()
