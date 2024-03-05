import os
import tempfile
from typing import List, Union

import streamlit as st
import tiktoken
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.text_splitter import (
    TextSplitter as LCSplitter,
)
from langchain.text_splitter import TokenTextSplitter as LCTokenTextSplitter
from llama_index import SimpleDirectoryReader
from llama_index.node_parser.interface import TextSplitter
from llama_index.schema import Document
from llama_index.text_splitter import CodeSplitter, SentenceSplitter, TokenTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."

text = st.sidebar.text_area("Enter text", value=DEFAULT_TEXT)
uploaded_files = st.sidebar.file_uploader("Upload file", accept_multiple_files=True)
type = st.sidebar.radio("Document Type", options=["Text", "Code"])
n_cols = st.sidebar.number_input("Columns", value=2, min_value=1, max_value=3)
assert isinstance(n_cols, int)


@st.cache_resource(ttl=3600)
def load_document(uploaded_files: List[UploadedFile]) -> List[Document]:
    # Read documents
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

    reader = SimpleDirectoryReader(input_dir=temp_dir.name)
    return reader.load_data()


if uploaded_files:
    if text != DEFAULT_TEXT:
        st.warning("Text will be ignored when uploading files")
    docs = load_document(uploaded_files)
    text = "\n".join([doc.text for doc in docs])


chunk_size = st.slider(
    "Chunk Size",
    value=512,
    min_value=1,
    max_value=4096,
)
chunk_overlap = st.slider(
    "Chunk Overlap",
    value=0,
    min_value=0,
    max_value=4096,
)

cols = st.columns(n_cols)
for ind, col in enumerate(cols):
    if type == "Text":
        text_splitter_cls = col.selectbox(
            "Text Splitter",
            options=[
                "TokenTextSplitter",
                "SentenceSplitter",
                "LC:RecursiveCharacterTextSplitter",
                "LC:CharacterTextSplitter",
                "LC:TokenTextSplitter",
            ],
            index=ind,
            key=f"splitter_cls_{ind}",
        )

        text_splitter: Union[TextSplitter, LCSplitter]
        if text_splitter_cls == "TokenTextSplitter":
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif text_splitter_cls == "SentenceSplitter":
            text_splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif text_splitter_cls == "LC:RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif text_splitter_cls == "LC:CharacterTextSplitter":
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif text_splitter_cls == "LC:TokenTextSplitter":
            text_splitter = LCTokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError("Unknown text splitter")
    elif type == "Code":
        text_splitter_cls = col.selectbox("Text Splitter", options=["CodeSplitter"])
        if text_splitter_cls == "CodeSplitter":
            language = col.text_input("Language", value="python")
            max_chars = col.slider("Max Chars", value=1500)

            text_splitter = CodeSplitter(language=language, max_chars=max_chars)
        else:
            raise ValueError("Unknown text splitter")

    chunks = text_splitter.split_text(text)
    tokenizer = tiktoken.get_encoding("gpt2").encode

    for chunk_ind, chunk in enumerate(chunks):
        n_tokens = len(tokenizer(chunk))
        n_chars = len(chunk)
        col.text_area(
            f"Chunk {chunk_ind} - {n_tokens} tokens - {n_chars} chars",
            chunk,
            key=f"text_area_{ind}_{chunk_ind}",
            height=500,
        )