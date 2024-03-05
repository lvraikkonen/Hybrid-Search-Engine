from typing import List, Dict, Any
from llama_index.schema import Document
from llama_index.node_parser import (
    SentenceWindowNodeParser,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.postprocessor import MetadataReplacementPostProcessor

from openai import OpenAI


class SentenceWindowRetriever:
    """
    Sentence Window Retriever

    Build input nodes from a text file by inserting metadata,
    build a vector index over the input nodes,
    then after retrieval insert the text into the output nodes
    before synthesis.

    """
    def __init__(self,
                 docs: List[Document] = None,
                 **kwargs: Any,
    ) -> None:
        # create the sentence window node parser w/ default settings
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
        )
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )
        # extract nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        self.sentence_index = VectorStoreIndex(
            nodes, service_context=self.service_context
        )
        self.postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        self.query_engine = self.sentence_index.as_query_engine(
            similarity_top_k=2,
            # the target key defaults to `window` to match the node_parser's default
            node_postprocessors=[self.postprocessor],
        )
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
