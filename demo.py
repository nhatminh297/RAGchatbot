import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.chroma import Chroma 
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.agents import Tool
from langchain.utilities.searchapi import SearchApiAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.document_loaders import WebBaseLoader

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
# os.environ["LANGCHAIN_API_KEY"] = 

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

def getQAtool(paths):
    template = """
    you are an expert about the topic of document they give to you.
    If they ask some common communication questions, you can answer them normally. 
    If they ask about knowledge or topics that you think it unrelate to the context provided, 
    remind them that you can only answer questions related to the topic of the document. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use five sentences maximum. Keep the answer as concise as possible. 
    If they are not polite to you, you have the right to be rude to them.
    {context}
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate.from_template(template)
        
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="full_documents", 
        embedding_function=GPT4AllEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    retriever.add_documents(get_docs(paths))
    # loader = WebBaseLoader("https://openai.com/pricing")
    # docs = loader.load()
    # retriever.add_documents(docs)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    with st.sidebar:
        st.text("You are free to ask now!")
    return qa

st.title("RAG ChatBot")

paths = []
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    
    st.subheader("Your documents")
    uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if uploaded_files is not None:
        destination_path = ".\docs"
        for uploaded_file in uploaded_files:
            destination_file_path = os.path.join(destination_path, uploaded_file.name)
            paths.append(destination_file_path)
            with open(destination_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
                
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

search = SearchApiAPIWrapper()
try:
    tools = [
    Tool(
        name="User uploaded documents QA system",
        func=getQAtool(paths),
        description="You should use this to aswer any of their question.",
    ),
    # Tool(
    #     name="Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events"
    # )
]
    # Construct the agent. We will use the default agent type here.
    # See documentation for a full list of options.
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
except Exception:
    st.warning('⚠ Please upload your Documents')


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
    st.secrets.openai_key = openai_api_key
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

        