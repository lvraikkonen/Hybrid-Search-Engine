# Hybrid-Search-Engine





## Get Start



1. Docker-Compose 启动Elasticsearch、Kibana，与Milvus向量数据库
2. 初始化ES中Index：`docs_qa`与向量存储Collection: `docs_qa`
3. Python项目依赖安装 `pip install -r requirements.txt`
4. 启动Web API
5. 可视化UI页面启动



## Features

### 功能支持

- [x] 文档问答API接口

- [ ] 文档上传

- [ ] 可视化UI

### 模型支持

Embedding Model

- [x] text-embedding-ada-002
- [ ] BAAI/bge-large-zh-v1.5

LLM

- [x] GPT-3.5-turbo
- [x] GPT-4
- [ ] ChatGLM4



###  文件格式支持

- [x] txt
- [x] pdf
- [ ] url
- [ ] markdown
- [ ] word



### RAG 检索阶段优化

- [x] Hybrid Search 关键词检索与向量检索集成
- [ ] Graph Store 知识图谱集成
- [ ] RAG-fusion RRF重排序
- [x] Rerank模型精排
- [ ] Embedding Fine-tune
- [x] 不同方法效果评估



## 结果总结与优化
