from typing import Dict, Any
import streamlit as st
from streamlit_pills import pills
import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from openai import OpenAI

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
                {"role": "assistant", "content": "Ask me a question:"}
            ]

        st.title(
            f"Chat with Your Document, powered by LlamaIndex 💬🦙"
        )
        st.info(
            "This example is powered by the **[Llama Hub Wikipedia Loader](https://llamahub.ai/l/wikipedia)**. Use any of [Llama Hub's many loaders](https://llamahub.ai/) to retrieve and chat with your data via a Streamlit app.",
            icon="ℹ️",
        )

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  # Add response to message history

        # get query_engine
        query_engine = DocQA()

        # custom_bm25_retriever = CustomBM25Retriever(top_k=3)
        # query = "给你机会你也不中用啊"
        # t_result = custom_bm25_retriever.retrieve(str_or_query_bundle=query)

        selected = pills(
            "Choose a question to get started or write your own below.",
            [
                "What is Snowflake?",
                "What company did Snowflake announce they would acquire in October 2023?",
                "What company did Snowflake acquire in March 2022?",
                "When did Snowflake IPO?",
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
                response = st.session_state["chat_engine"].answer(selected)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                add_to_message_history("user", selected)
                add_to_message_history("assistant", response)

        if prompt := st.chat_input(
            "Your question"
        ):  # Prompt for user input and save to chat history
            add_to_message_history("user", prompt)

            # Display the new question immediately after it is entered
            with st.chat_message("user"):
                st.write(prompt)

            # If last message is not from assistant, generate a new response
            # if st.session_state["messages"][-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = st.session_state["chat_engine"].answer(prompt)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                # st.write(response.response)
                add_to_message_history("assistant", response.response)

            # Save the state of the generator
            st.session_state["response_gen"] = response.response_gen


if __name__ == "__main__":
    StreamlitChat(run_from_main=True).run()