import os
import sys
import tempfile
import subprocess
import streamlit as st

# ========== 🧪 Debugging Utilities ==========
def debug_environment():
    import platform
    import pkg_resources

    st.subheader("🧪 Debug Info")
    st.write("🐍 Python:", sys.version)
    st.write("📁 Python Path:", sys.executable)
    st.write("📦 Virtual Env:", sys.prefix)
    try:
        import langchain_community
        st.success("✅ langchain_community is installed.")
    except ImportError:
        st.error("❌ langchain_community is NOT installed.")
    try:
        result = subprocess.run(["pip", "list"], capture_output=True, text=True)
        st.code(result.stdout)
    except Exception as e:
        st.error(f"Couldn't list packages: {e}")

st.sidebar.button("🧪 Show Debug Info", on_click=debug_environment)

# ========== ✅ Try imports ==========
try:
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import RetrievalQA
except ModuleNotFoundError as e:
    st.error(f"🚨 Missing Module: {e.name}")
    st.stop()

# ========== 💻 Streamlit UI ==========
st.title("📄 Chat with Your Document")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📤 Upload PDF or Word document", type=["pdf", "docx"])
user_query = st.text_input("💬 Ask a question about the document:")

if uploaded_file and openai_api_key and user_query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load document: {e}")
        st.stop()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed and store in Chroma
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        st.stop()

    # Build retrieval chain
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-0613")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        response = qa_chain.run(user_query)
        st.success("🧠 Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
