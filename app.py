import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("📄 Chat with Your Document")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📤 Upload PDF or Word document", type=["pdf", "docx"])
user_query = st.text_input("💬 Ask a question about the document:")

if uploaded_file and openai_api_key and user_query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyMuPDFLoader(tmp_path)
    else:
        loader = Docx2txtLoader(tmp_path)

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed and store in Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

    # Create retrieval chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-0613")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Run QA
    response = qa_chain.run(user_query)
    st.write("🧠 Answer:", response)
