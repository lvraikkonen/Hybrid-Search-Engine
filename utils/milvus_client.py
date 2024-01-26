from utils.logger import logger
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    Partition
)

import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from config.config_parser import (
    MILVUS_HOST_LOCAL, MILVUS_HOST_ONLINE, MILVUS_PORT,
    MILVUS_SIZE, MILVUS_THRESHOLD
)

from langchain.docstore.document import Document
from typing import List


class MilvusFailed(Exception):
    """异常基类"""
    pass



class MilvusClient:
    def __init__(self, mode, collection_name, *, threshold=1.1, client_timeout=3):
        # self.user_id = user_id
        # self.kb_ids = kb_ids
        self.collection_name = collection_name
        self.logger = logger
        if mode == 'local':
            self.host = MILVUS_HOST_LOCAL
        else:
            self.host = MILVUS_HOST_ONLINE
        self.port = MILVUS_PORT
        self.client_timeout = client_timeout
        self.threshold = threshold
        self.sess: Collection = None
        # self.partitions: List[Partition] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.top_k = MILVUS_SIZE
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 256}}
        if mode == 'local':
            self.create_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        else:
            self.create_params = {"metric_type": "L2", "index_type": "GPU_IVF_FLAT", "params": {"nlist": 2048}}
        self.last_init_ts = time.time() - 100  # 减去100保证最初的init不会被拒绝
        self.init()
    
    @property
    def fields(self):
        fields = [
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name='file_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=640, description="source file name"),
            FieldSchema(name='file_path', dtype=DataType.VARCHAR, max_length=640, description="source file path"),
            FieldSchema(name='timestamp', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=2000, description="raw text"),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536,
                    description="store embedding of text array")
        ]
        return fields

    def parse_batch_result(self, batch_result):
        new_result = []
        for batch_idx, result in enumerate(batch_result):
            new_cands = []
            result.sort(key=lambda x: x.score)
            valid_results = [cand for cand in result if cand.score <= self.threshold]
            if len(valid_results) == 0:  # 如果没有合适的结果，就取topk
                valid_results = result[:self.top_k]
            for cand_i, cand in enumerate(valid_results):
                doc = Document(page_content=cand.entity.get('content'),
                               metadata={"score": cand.score, "file_id": cand.entity.get('file_id'),
                                         "file_name": cand.entity.get('file_name'),
                                         "chunk_id": cand.entity.get('chunk_id')})
                new_cands.append(doc)
            new_cands = self.expand_cand_docs(new_cands)
            new_result.append(new_cands)
        return new_result

    @property
    def output_fields(self):
        return ['chunk_id', 'file_id', 'file_name', 'file_path', 'timestamp', 'content']

    def init(self):
        try:
            connections.connect(host=self.host, port=self.port)  # timeout=3 [cannot set]
            if utility.has_collection(self.collection_name):
                self.sess = Collection(self.collection_name)
                self.logger.info(f'collection {self.collection_name} exists')
            else:
                schema = CollectionSchema(self.fields, description="vector db for docs qa")
                self.logger.info(f'create collection {self.collection_name} {schema}')
                self.sess = Collection(self.collection_name, schema)
                self.sess.create_index(field_name="embeddings", index_params=self.create_params)
            # for kb_id in self.kb_ids:
            #     if not self.sess.has_partition(kb_id):
            #         self.sess.create_partition(kb_id)
            # self.partitions = [Partition(self.sess, kb_id) for kb_id in self.kb_ids]
            # self.logger.info('partitions: %s', self.kb_ids)
            self.sess.load()
        except Exception as e:
            self.logger.error(e)

    def __search_emb_sync(self, embs, expr='', top_k=None, client_timeout=None):
        if not top_k:
            top_k = self.top_k
        milvus_records = self.sess.search(data=embs, anns_field="embeddings",
                                          param=self.search_params, limit=top_k,
                                          output_fields=self.output_fields, expr=expr, timeout=client_timeout)
        # self.logger.info(milvus_records)
        return self.parse_batch_result(milvus_records)

    def search_emb_async(self, embs, expr='', top_k=None, client_timeout=None):
        if not top_k:
            top_k = self.top_k
        # 将search_emb_sync函数放入线程池中运行
        future = self.executor.submit(self.__search_emb_sync, embs, expr, top_k, client_timeout)
        return future.result()

    def query_expr_async(self, expr, output_fields=None, client_timeout=None):
        if client_timeout is None:
            client_timeout = self.client_timeout
        if not output_fields:
            output_fields = self.output_fields
        future = self.executor.submit(
            partial(self.sess.query, output_fields=output_fields, expr=expr,
                    timeout=client_timeout))
        return future.result()

    async def insert_files(self, file_id, file_name, file_path, docs, embs, batch_size=1000):
        self.logger.info(f'now inser_file {file_name}')
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        loop = asyncio.get_running_loop()
        contents = [doc.page_content for doc in docs]
        num_docs = len(docs)
        for batch_start in range(0, num_docs, batch_size):
            batch_end = min(batch_start + batch_size, num_docs)
            data = [[] for _ in range(len(self.sess.schema))]

            for idx in range(batch_start, batch_end):
                cont = contents[idx]
                emb = embs[idx]
                chunk_id = f'{file_id}_{idx}'
                data[0].append(chunk_id)
                data[1].append(file_id)
                data[2].append(file_name)
                data[3].append(file_path)
                data[4].append(timestamp)
                data[5].append(cont)
                data[6].append(emb)

            # 执行插入操作
            try:
                self.logger.info('Inserting into Milvus...')
                mr = await loop.run_in_executor(
                    self.executor, partial(self.sess.insert, data=data))
                self.logger.info(f'{file_name} {mr}')
            except Exception as e:
                self.logger.info(f'Milvus insert file_id:{file_id}, file_name:{file_name} failed: {e}')
                return False

        return True

    def delete_collection(self):
        self.sess.release()
        utility.drop_collection(self.collection_name)

    def delete_partition(self, partition_name):
        part = Partition(self.sess, partition_name)
        part.release()
        self.sess.drop_partition(partition_name)

    def delete_files(self, files_id):
        self.sess.delete(expr=f"file_id in {files_id}")
        self.logger.info('milvus delete files_id: %s', files_id)

    def get_files(self, files_id):
        res = self.query_expr_async(expr=f"file_id in {files_id}", output_fields=["file_id"])
        valid_ids = [result['file_id'] for result in res]
        return valid_ids

    # def seperate_list(self, ls: List[int]) -> List[List[int]]:
    #     lists = []
    #     ls1 = [ls[0]]
    #     for i in range(1, len(ls)):
    #         if ls[i - 1] + 1 == ls[i]:
    #             ls1.append(ls[i])
    #         else:
    #             lists.append(ls1)
    #             ls1 = [ls[i]]
    #     lists.append(ls1)
    #     return lists

    # def process_group(self, group):
    #     new_cands = []
    #     group.sort(key=lambda x: int(x.metadata['chunk_id'].split('_')[-1]))
    #     id_set = set()
    #     file_id = group[0].metadata['file_id']
    #     file_name = group[0].metadata['file_name']
    #     group_scores_map = {}
    #     # 先找出该文件所有需要搜索的chunk_id
    #     cand_chunks = []
    #     for cand_doc in group:
    #         current_chunk_id = int(cand_doc.metadata['chunk_id'].split('_')[-1])
    #         group_scores_map[current_chunk_id] = cand_doc.metadata['score']
    #         for i in range(current_chunk_id - 200, current_chunk_id + 200):
    #             need_search_id = file_id + '_' + str(i)
    #             if need_search_id not in cand_chunks:
    #                 cand_chunks.append(need_search_id)

    #     group_relative_chunks = self.query_expr_async(expr=f"file_id == \"{file_id}\" and chunk_id in {cand_chunks}",
    #                                                   output_fields=["chunk_id", "content"])
    #     group_chunk_map = {int(item['chunk_id'].split('_')[-1]): item['content'] for item in group_relative_chunks}
    #     group_file_chunk_num = list(group_chunk_map.keys())
    #     for cand_doc in group:
    #         current_chunk_id = int(cand_doc.metadata['chunk_id'].split('_')[-1])
    #         doc = copy.deepcopy(cand_doc)
    #         id_set.add(current_chunk_id)
    #         docs_len = len(doc.page_content)
    #         for k in range(1, 200):
    #             break_flag = False
    #             for expand_index in [current_chunk_id + k, current_chunk_id - k]:
    #                 if expand_index in group_file_chunk_num:
    #                     merge_content = group_chunk_map[expand_index]
    #                     if docs_len + len(merge_content) > CHUNK_SIZE:
    #                         break_flag = True
    #                         break
    #                     else:
    #                         docs_len += len(merge_content)
    #                         id_set.add(expand_index)
    #             if break_flag:
    #                 break

    #     id_list = sorted(list(id_set))
    #     id_lists = self.seperate_list(id_list)
    #     for id_seq in id_lists:
    #         for id in id_seq:
    #             if id == id_seq[0]:
    #                 doc = Document(page_content=group_chunk_map[id],
    #                                metadata={"score": 0, "file_id": file_id,
    #                                          "file_name": file_name})
    #             else:
    #                 doc.page_content += " " + group_chunk_map[id]
    #         doc_score = min([group_scores_map[id] for id in id_seq if id in group_scores_map])
    #         doc.metadata["score"] = format(1 - doc_score / math.sqrt(2), '.4f')
    #         doc.metadata["kernel"] = '|'.join([group_chunk_map[id] for id in id_seq if id in group_scores_map])
    #         new_cands.append(doc)
    #     return new_cands

    # def expand_cand_docs(self, cand_docs):
    #     cand_docs = sorted(cand_docs, key=lambda x: x.metadata['file_id'])
    #     # 按照file_id进行分组
    #     m_grouped = [list(group) for key, group in groupby(cand_docs, key=lambda x: x.metadata['file_id'])]
    #     self.logger.info('milvus group number: %s', len(m_grouped))

    #     with ThreadPoolExecutor(max_workers=10) as executor:
    #         futures = []
    #         # 对每个分组按照chunk_id进行排序
    #         for group in m_grouped:
    #             if not group:
    #                 continue
    #             future = executor.submit(self.process_group, group)
    #             futures.append(future)

    #         new_cands = []
    #         for future in as_completed(futures):
    #             result = future.result()
    #             if result is not None:
    #                 new_cands.extend(result)
    #         return new_cands


if __name__ == '__main__':
    client = MilvusClient(mode='local', collection_name="unit_test_collection")