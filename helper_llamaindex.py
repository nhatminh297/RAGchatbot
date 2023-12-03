
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import load_index_from_storage
from llama_index import SimpleDirectoryReader
from llama_index import Document
import os


def build_sentence_window_index(
    paths,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="./llama_index",
):
    
    documents = SimpleDirectoryReader(input_files=paths).load_data()
    document = [Document(text="\n\n".join([doc.text for doc in documents]))]

    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        system_prompt="""Based on the provided documents and information, 
        You are an expert on the topics of all documents that user uploaded.
        Your job is to answer questions related to information about their documents.
        Keep your answers within a maximum of 7 sentences.
        Keep the response as concise as posible and based on facts – do not hallucinate features.
        If you don't know the answer, remind them what topics you can answer to.
        """

    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            document, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_chat_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=3
):
    # define postprocessors
    # postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    # rerank = SentenceTransformerRerank(
    #     top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    # )

    sentence_window_chat_engine = sentence_index.as_chat_engine(chat_mode="condense_question",
                                                                verbose=True)
    return sentence_window_chat_engine
