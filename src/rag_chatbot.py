import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os
import tempfile
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np

# Set page configuration
st.set_page_config(page_title="Ollama RAG Chatbot", page_icon="🤖", layout="wide")

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight embedding model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 3

# Initialize embedding model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Initialize Chroma DB
@st.cache_resource
def get_vector_store():
    # Use a persistent directory if you want to save the database
    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    
    embeddings = get_embeddings_model()
    # Return empty DB if first time
    try:
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    except:
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Function to get available models
def get_available_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            models = [model["name"] for model in models_data]
            return models
        else:
            st.sidebar.error(f"Failed to get models: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Connection error: {e}. Is Ollama running?")
        return []

# Function to process documents and add to vector store
def process_document(uploaded_file, vector_store):
    # Create a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    # Save uploaded file to temp location
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Determine loader based on file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_extension in [".txt", ".py", ".js", ".html", ".css", ".json"]:
            loader = TextLoader(file_path)
        else:
            return f"Unsupported file type: {file_extension}"
        
        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add document source metadata
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
        
        # Add to vector store
        vector_store.add_documents(chunks)
        
        return f"Successfully processed {uploaded_file.name} into {len(chunks)} chunks"
    except Exception as e:
        return f"Error processing document: {str(e)}"
    finally:
        temp_dir.cleanup()

# Function to retrieve relevant documents
def retrieve_context(query, vector_store, k=TOP_K_RETRIEVAL):
    try:
        results = vector_store.similarity_search(query, k=k)
        contexts = []
        
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            contexts.append(f"From {source}:\n{doc.page_content}")
        
        return "\n\n".join(contexts)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# Function to call Ollama API with retrieved context
def call_ollama_with_rag(prompt: str, model: str, retrieved_context: str) -> Dict[str, Any]:
    messages = []
    
    # Add system prompt with RAG context
    if retrieved_context:
        system_message = f"{st.session_state.system_prompt}\n\nRelevant context for answering:\n{retrieved_context}"
    else:
        system_message = st.session_state.system_prompt
    
    messages.append({"role": "system", "content": system_message})
    
    # Add chat history 
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": st.session_state.temperature}
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {e}"}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant. Use the context provided to answer questions."
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "show_retrieved_context" not in st.session_state:
    st.session_state.show_retrieved_context = False

# Sidebar setup
st.sidebar.title("Ollama RAG Chatbot")

# Get available models
models = get_available_models()
if not models:
    st.sidebar.warning("No models found or Ollama is not running. Please start Ollama and refresh.")
    selected_model = ""
else:
    selected_model = st.sidebar.selectbox("Select a model", models)

# Temperature slider
st.session_state.temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=2.0, 
    value=st.session_state.temperature, 
    step=0.1,
    help="Higher values make output more random, lower values more deterministic"
)

# System prompt
st.session_state.system_prompt = st.sidebar.text_area(
    "System Prompt", 
    value=st.session_state.system_prompt, 
    help="Instructions for the AI's behavior"
)

# RAG settings
st.sidebar.subheader("Knowledge Base Settings")
st.session_state.rag_enabled = st.sidebar.checkbox("Enable RAG", value=st.session_state.rag_enabled)
st.session_state.show_retrieved_context = st.sidebar.checkbox("Show Retrieved Context", value=st.session_state.show_retrieved_context)

# File uploader for knowledge base
st.sidebar.subheader("Add to Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", 
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "md", "py", "js", "html", "css", "json"]
)

# Process uploaded files
if uploaded_files:
    vector_store = get_vector_store()
    for uploaded_file in uploaded_files:
        with st.sidebar.status(f"Processing {uploaded_file.name}..."):
            result = process_document(uploaded_file, vector_store)
            st.sidebar.write(result)

# Main chat interface
st.title("💬 Ollama RAG Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    if not selected_model:
        st.error("Please select a model from the sidebar first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get relevant context if RAG is enabled
        retrieved_context = ""
        if st.session_state.rag_enabled:
            vector_store = get_vector_store()
            if vector_store._collection.count() > 0:  # Check if there are documents in the store
                with st.status("Retrieving context..."):
                    retrieved_context = retrieve_context(prompt, vector_store)
        
        # Show retrieved context if enabled
        if st.session_state.show_retrieved_context and retrieved_context:
            with st.expander("Retrieved Context"):
                st.markdown(retrieved_context)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.rag_enabled:
                    response = call_ollama_with_rag(prompt, selected_model, retrieved_context)
                else:
                    response = call_ollama_with_rag(prompt, selected_model, "")
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    assistant_response = response.get("message", {}).get("content", "No response")
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Clear chat button and knowledge base stats
st.sidebar.divider()
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Show knowledge base stats
try:
    vector_store = get_vector_store()
    document_count = vector_store._collection.count()
    st.sidebar.write(f"Knowledge base documents: {document_count}")
except:
    st.sidebar.write("Knowledge base not initialized")

if st.sidebar.button("Clear Knowledge Base"):
    try:
        vector_store = get_vector_store()
        vector_store._collection.delete(vector_store._collection.get()["ids"])
        st.sidebar.success("Knowledge base cleared")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing knowledge base: {str(e)}")

# Footer
st.sidebar.divider()
st.sidebar.markdown("Built with Streamlit + Ollama + RAG")