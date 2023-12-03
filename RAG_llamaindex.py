import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from langchain.embeddings import GPT4AllEmbeddings
from helper_llamaindex import *

st.set_page_config(page_title="Chat with your docs LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("RAG with LlamaIndex 💬🦙")
         
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

def load_data(paths: list):
    if not os.path.exists("./llama_index"):
        documents = SimpleDirectoryReader(input_files=paths).load_data()    
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
                                                                system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir="./llama_index")
    else:
        storage_context = StorageContext.from_defaults(persist_dir="./llama_index")
        index = load_index_from_storage(storage_context)
    return index

paths = []
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # upload files
    st.subheader("Your documents")
    uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if uploaded_files is not None:
        destination_path = ".\docs"
        paths.append(os.path.join(destination_path, "temp.txt"))
        
        for uploaded_file in uploaded_files:
            destination_file_path = os.path.join(destination_path, uploaded_file.name)
            paths.append(destination_file_path)
            with open(destination_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

        if st.button("Process"):
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
                
            openai.api_key = openai_api_key
            with st.spinner(text="Loading and indexing docs! This should take 1-2 minutes."):
                index = build_sentence_window_index(paths, 
                                                    OpenAI(model="gpt-3.5-turbo", temperature=0.1),
                                                    save_dir="./llama_index")

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

index = build_sentence_window_index(paths, 
                                    OpenAI(model="gpt-3.5-turbo", temperature=0.1),
                                    save_dir="./llama_index")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = get_sentence_window_chat_engine(index, similarity_top_k=6)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)