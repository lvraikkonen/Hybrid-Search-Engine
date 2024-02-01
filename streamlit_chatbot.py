from typing import Dict, Any
from utils.logger import logger
import streamlit as st
from streamlit_pills import pills
from config.config_parser import SYSTEM_ROLE
from faiss import IndexFlatIP

from llama_index.query_engine import RetrieverQueryEngine
from openai import OpenAI
from custom_retrievers.ensemble_rerank_retirever import EnsembleRerankRetriever
from custom_retrievers.ensemble_retriever import EnsembleRetriever
from custom_retrievers.vector_store_retriever import VectorSearchRetriever

# from custom_retrievers.bm25_retriever import CustomBM25Retriever
from Doc_QA import DocQA


class StreamlitChat:
    """Streamlit chatbot."""

    def __init__(
        self,
        # wikipedia_page: str = "Snowflake Inc.",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        pass
        # self.wikipedia_page = wikipedia_page

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""

        st.set_page_config(
            page_title=f"Chat with Your Document, powered by LlamaIndex",
            page_icon="🦙",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:  # Initialize the chat messages history
            st.session_state["messages"] = [
                {"role": "system", "content": SYSTEM_ROLE}
            ]

        st.title(
            f"Chat with Your Document, powered by LlamaIndex 💬🦙"
        )
        st.info(
            "This example is powered by the Elasticsearch and Milvus."
            ".Retrieve and Chat with your data via a Streamlit app.",
            icon="ℹ️",
        )

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  # Add response to message history
            # logger.info(f"add prompt from {role} with message: {content}")

        faiss_index = IndexFlatIP(1536)
        # retriever = EnsembleRerankRetriever(top_k=2, faiss_index=faiss_index)
        retriever = VectorSearchRetriever(top_k=3, faiss_index=faiss_index)
        query_engine = RetrieverQueryEngine.from_args(retriever)

        # custom_bm25_retriever = CustomBM25Retriever(top_k=3)
        # query = "给你机会你也不中用啊"
        # t_result = custom_bm25_retriever.retrieve(str_or_query_bundle=query)

        selected = pills(
            "Choose a question to get started or write your own below.",
            [
                "中国队亚洲杯成绩如何?",
                "中国队能小组出线吗?",
                "中国男足主教练是谁?",
            ],
            clearable=True,
            index=None,
        )

        if "chat_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["chat_engine"] = query_engine

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

            # To avoid duplicated display of answered pill questions each rerun
        if selected and selected not in st.session_state.get(
                "displayed_pill_questions", set()
        ):
            st.session_state.setdefault("displayed_pill_questions", set()).add(selected)
            with st.chat_message("user"):
                st.write(selected)
            with st.chat_message("assistant"):
                with st.spinner('Query is at work...'):
                    response = st.session_state["chat_engine"].query(selected)
                    st.markdown(response)
                    add_to_message_history("user", selected)
                    add_to_message_history("assistant", response)

        if prompt := st.chat_input(
                "Your question"
        ):  # Prompt for user input and save to chat history
            add_to_message_history("user", prompt)

            # Display the new question immediately after it is entered
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner('Query is at work...'):
                    response = st.session_state["chat_engine"].query(prompt)
                    logger.info(f"response text is: {response}")
                    st.markdown(response)
                    add_to_message_history("assistant", response)


if __name__ == "__main__":
    StreamlitChat(run_from_main=True).run()
