import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
from huggingface_hub import login
import streamlit as st
import tempfile
from PyPDF2 import PdfReader

# Hugging Face Login
login(token=st.secrets["HUGGINGFACE_API_KEY"])

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

# Pinecone Initialization
pinecone_index_name = "rag-chat-index"
pinecone_client = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

if pinecone_index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pinecone_client.Index(pinecone_index_name)

# Hugging Face Model Setup
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.8,
    top_k=50,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
)
embeddings = HuggingFaceEmbeddings()

# Streamlit UI Setup
st.set_page_config(page_title="RAG Chat Application")
st.title("RAG Chat Application")
st.sidebar.header("Upload Documents")

# File Upload and Processing
docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if docs:
    all_text = ""
    for doc in docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc.read())
            pdf_reader = PdfReader(temp_file.name)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            all_text += text

    # Split and Embed Documents
    st.sidebar.write("Indexing documents...")
    splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(all_text)
    vectors = embeddings.embed_documents(chunks)
    metadata = [{"text": chunk} for chunk in chunks]

    for vector, meta in zip(vectors, metadata):
        pinecone_index.upsert([(str(hash(meta["text"])), vector, meta)])

    st.sidebar.success("Documents indexed successfully!")

# Define Prompt Template
template = """
You are a chatbot for the Government. Corporation workers will ask questions regarding the procedures for uploading documents. 
Answer these questions and give answers to process in a step-by-step process.
If you don't know the answer, just say you don't know. 

Context: {context}
Question: {question}
Answer: 
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Chat Functionality
st.header("Ask Questions")
user_query = st.text_input("Enter your question:")
if user_query:
    # Retrieve Relevant Documents
    query_embedding = embeddings.embed_query(user_query)
    results = pinecone_index.query(query_embedding, top_k=5, include_metadata=True)
    relevant_texts = [item["metadata"]["text"] for item in results["matches"]]

    # Generate Response
    context = "\n".join(relevant_texts)
    query_prompt = prompt.format(context=context, question=user_query)
    response = llm.invoke(query_prompt)

    # Display Response
    st.write("### Response:")
    st.write(response)
