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
from langchain.vectorstores import Chroma
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


def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150, 
        separators=["\n\n", "\n", "(?<=\. )", " ", ""], 
        length_function = len
        )
    splits = text_splitter.split_documents(docs)
    return splits


def getAgent(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return agent

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
                splits = get_text_chunks(raw_text)
                embedding = OpenAIEmbeddings()
                persist_dir = "./chroma"

                vectordb = Chroma.from_documents(documents=splits,
                                                embedding=embedding,
                                                persist_directory=persist_dir)

vectordb2 = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma")
vectordb2.get()
agent = getAgent(vectordb2)

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

        