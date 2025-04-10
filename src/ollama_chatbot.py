import streamlit as st
import requests
from typing import List, Dict, Any

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="wide")

# Sidebar for model selection
st.sidebar.title("Ollama Chatbot")
OLLAMA_API_BASE = "http://localhost:11434/api"

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

# Get available models
models = get_available_models()

if not models:
    st.sidebar.warning("No models found or Ollama is not running. Please start Ollama and refresh.")
    selected_model = ""
else:
    selected_model = st.sidebar.selectbox("Select a model", models)

# Temperature slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1,
                              help="Higher values make output more random, lower values more deterministic")

# System prompt
system_prompt = st.sidebar.text_area("System Prompt (Optional)", 
                                  value="You are a helpful AI assistant.", 
                                  help="Instructions for the AI's behavior")

# Main chat interface
st.title("💬 Chat with Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to call Ollama API
def call_ollama(prompt: str, model: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}]
    
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
                "options": {"temperature": temperature}
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {e}"}

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
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_ollama(prompt, selected_model)
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    assistant_response = response.get("message", {}).get("content", "No response")
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Show current settings
st.sidebar.divider()
st.sidebar.subheader("Current Settings")
st.sidebar.write(f"Model: {selected_model if selected_model else 'None selected'}")
st.sidebar.write(f"Temperature: {temperature}")

# Footer
st.sidebar.divider()
