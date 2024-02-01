## 现有市面上的分块工具

- [ ] CharacterTextSplitter
- [ ] LatexTextSplitter
- [ ] MarkdownHeaderTextSplitter
- [ ] MarkdownTextSplitter
- [ ] NLTKTextSplitter
- [ ] PythonCodeTextSplitter
- [ ] RecursiveCharacterTextSplitter
- [ ] SentenceTransformersTokenTextSplitter
- [ ] SpacyTextSplitter
- [ ] AliTextSplitter
- [ ] ChineseRecursiveTextSplitter
- [ ] ChineseTextSplitter

## 分块方式

1. 固定大小分块

决定块中的tokens的数量，以及它们之间是否应该有任何重叠。一般来说，我们会在块之间保持一些重叠，以确保语义上下文不会在块之间丢失。在大多数情况下，固定大小的分块将是最佳方式。

``` python
text = "..." # 你的文本
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # 设置一个非常小的块大小。
    chunk_size = 256,
    chunk_overlap  = 20
)

docs = text_splitter.create_documents([text])
```

2. 基于内容意图分割


### 1. RecursiveCharacterTextSplitter

``` python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.create_documents([fullDoc])
for i, text in enumerate(texts):
    print(f'doc: #{i}', text)
```

### 2. SpacyTextSplitter

加载中文库
``` shell
python -m spacy download zh_core_web_sm
```

``` python
from langchain.text_splitter import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_text(fullDoc)
for i, text in enumerate(texts):
    print(f'doc: #{i}', text)
```

### 3. NLTKTextSplitter


## Specialized chunking

Markdown and LaTeX

### Markdown

``` python
from langchain.text_splitter import MarkdownTextSplitter
markdown_text = "..."

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])
```

### LaTeX

``` python
from langchain.text_splitter import LatexTextSplitter
latex_text = "..."
latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
docs = latex_splitter.create_documents([latex_text])
```


## Multi-Modal Chunking

Mixture of Text, Image, Table