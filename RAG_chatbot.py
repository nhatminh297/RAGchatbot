import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain import hub
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentExecutor
import random
import time



def get_docs(paths):
    loaders=[]
    for path in paths:
        loaders.append(PyPDFLoader(path))
    docs=[]
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    nltk_splitter = NLTKTextSplitter(chunk_size=200)
    splits = nltk_splitter.split_documents(docs)
    return splits


def getAgent(vectorstore):
    prompt = hub.pull("rlm/rag-prompt")    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=prompt
    )
    return agent

def set_vector_store(docs, embed_model, save_dir):
    faiss_db = FAISS.from_documents(docs, embed_model)
    faiss_db.save_local(save_dir)

load_dotenv()
st.title("RAG ChatBot")

with st.sidebar:
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

faiss_db = FAISS.load_local("faiss_index", GPT4AllEmbeddings())

agent = getAgent(faiss_db)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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

        