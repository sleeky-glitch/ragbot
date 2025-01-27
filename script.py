import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from huggingface_hub import login
from transformers import pipeline
import pinecone
import os
import tempfile
from PyPDF2 import PdfReader

# Hugging Face login
login(token=st.secrets["huggingface_api_key"])

# Pinecone setup
pinecone.init(api_key=st.secrets["pinecone_api_key"])
index_name = "rag-chat-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")
pinecone_client = pinecone.Index(index_name)

# Model setup
model_name = "mistralai/Mixtral-8x7B-v0.1"
huggingface_pipeline = pipeline("text-generation", model=model_name, device=0)  # Use GPU if available
llm = HuggingFacePipeline(pipeline=huggingface_pipeline)

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
    vectors = [llm.pipeline.model.encode(chunk) for chunk in chunks]  # Use the model to encode chunks
    metadata = [{"text": chunk} for chunk in chunks]

    for vector, meta in zip(vectors, metadata):
        pinecone_client.upsert([(str(hash(meta["text"])), vector, meta)])

    st.sidebar.success("Documents indexed successfully!")

# Chat functionality
st.header("Ask Questions")
user_query = st.text_input("Enter your question:")
if user_query:
    # Retrieve relevant documents
    query_embedding = llm.pipeline.model.encode(user_query)  # Use the model to encode the query
    results = pinecone_client.query(query_embedding, top_k=5, include_metadata=True)
    relevant_texts = [item["metadata"]["text"] for item in results["matches"]]

    # Formulate context
    context = "\n".join(relevant_texts)
    prompt = PromptTemplate(template="Context: {context}\nQuestion: {query}\nAnswer:", variables=["context", "query"])
    query_prompt = prompt.format(context=context, query=user_query)

    # Get response from LLM
    response = llm.invoke(query_prompt)
    st.write("### Response:")
    st.write(response)
