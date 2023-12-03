import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.prompts import PromptTemplate
from langchain import hub
from openai import OpenAI


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "ls__87dc3360db3c42dab3bf3f911e4923f0"

def get_docs(paths):
    loaders=[]
    for path in paths:
        loaders.append(PyPDFLoader(path))
    docs=[]
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150, 
        separators=["\n\n", "\n", "(?<=\. )", " ", ""], 
        length_function = len
        )
    splits = text_splitter.split_documents(docs)
    return splits


def getAgent(vectorstore):
    template = """
    If they ask some common communication questions, you can answer them normally. 
    If they ask about knowledge or topics that you think it unrelate to the context provided, 
    remind them that you can only answer questions related to the topic of the document. 
    But you can answer about what is in the chat.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use five sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate.from_template(template)
        
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return agent

def set_vector_store(docs, embed_model, save_dir):
    faiss_db = FAISS.from_documents(docs, embed_model)
    faiss_db.save_local(save_dir)

load_dotenv()
st.title("RAG ChatBot")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    
    st.subheader("Your documents")
    uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if uploaded_files is not None:
        destination_path = ".\docs"
        paths = []
        for uploaded_file in uploaded_files:
            destination_file_path = os.path.join(destination_path, uploaded_file.name)
            paths.append(destination_file_path)
            with open(destination_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_docs(paths)
                docs = split_documents(raw_text)
                embedding = GPT4AllEmbeddings()
                set_vector_store(docs = docs, embed_model=embedding, save_dir='faiss_index')
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

faiss_db = FAISS.load_local("faiss_index", GPT4AllEmbeddings())

agent = getAgent(faiss_db)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input := st.chat_input("How may I assist you today?"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": input})

    with st.chat_message("user"):
        st.markdown(input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in agent.run(
                st.session_state.messages[-1]['content']
            ):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

        