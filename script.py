import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_index import GPTSimpleVectorIndex
from huggingface_hub import login
import pinecone
import os
import tempfile
from PyPDF2 import PdfReader

# Hugging Face login
login(token=st.secrets["huggingface_api_key"])

# Pinecone setup
pinecone.init(api_key=st.secrets["pinecone_api_key"], environment=st.secrets["pinecone_env"])
index_name = "rag-chat-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")
pinecone_client = pinecone.Index(index_name)

# Model setup
model_name = "mistralai/Mixtral-8x7B-v0.1"
llm = HuggingFaceLLM(model_name=model_name)
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Streamlit UI
st.set_page_config(page_title="RAG Chat Application")
st.title("RAG Chat Application")
st.sidebar.header("Upload Documents")

# File upload
docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if docs:
    all_text = ""
    for doc in docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc.read())
            pdf_reader = PdfReader(temp_file.name)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            all_text += text

    # Split and index documents
    st.sidebar.write("Indexing documents...")
    splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(all_text)
    vectors = [embeddings.embed_text(chunk) for chunk in chunks]
    metadata = [{"text": chunk} for chunk in chunks]

    for vector, meta in zip(vectors, metadata):
        pinecone_client.upsert([(str(hash(meta["text"])), vector, meta)])

    st.sidebar.success("Documents indexed successfully!")

# Chat functionality
st.header("Ask Questions")
user_query = st.text_input("Enter your question:")
if user_query:
    # Retrieve relevant documents
    results = pinecone_client.query(embeddings.embed_text(user_query), top_k=5, include_metadata=True)
    relevant_texts = [item["metadata"]["text"] for item in results["matches"]]

    # Formulate context
    context = "\n".join(relevant_texts)
    prompt = PromptTemplate(template="Context: {context}\nQuestion: {query}\nAnswer:", variables=["context", "query"])
    query_prompt = prompt.format(context=context, query=user_query)

    # Get response from LLM
    response = llm.invoke(query_prompt)
    st.write("### Response:")
    st.write(response)
