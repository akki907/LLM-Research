import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time
from helper import get_available_models,OLLAMA_API_BASE

# Set page configuration
st.set_page_config(page_title="Ollama Chatbot with Streaming", page_icon="🤖", layout="wide")

# Sidebar for model selection
st.sidebar.title("Ollama Chatbot")


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

# Stream speed slider
stream_speed = st.sidebar.slider("Stream Speed", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                              help="Adjust streaming speed (lower is faster)")

# Main chat interface
st.title("💬 Chat with Ollama (Streaming)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to stream Ollama API response
def stream_ollama_response(prompt: str, model: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history 
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Stream the response
        response = requests.post(
            f"{OLLAMA_API_BASE}/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,  # Enable streaming
                "options": {"temperature": temperature}
            },
            stream=True  # Enable streaming at the requests level
        )
        
        # Initialize variables for streaming
        full_response = ""
        placeholder = st.empty()
        
        # Process the streamed response
        for chunk in response.iter_lines():
            if chunk:
                try:
                    chunk_data = json.loads(chunk.decode("utf-8"))
                    
                    # Extract the message content from the chunk
                    if "message" in chunk_data:
                        message_content = chunk_data["message"].get("content", "")
                        if message_content:
                            full_response += message_content
                            placeholder.markdown(full_response + "▌")
                            
                            # Add a small delay based on stream speed setting
                            # Lower speed value means faster streaming
                            time.sleep(max(0.001, stream_speed * 0.02))
                    
                    # Check for done status
                    if chunk_data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        # Final update without the cursor
        placeholder.markdown(full_response)
        return full_response
        
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return f"Error: {str(e)}"

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
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            assistant_response = stream_ollama_response(prompt, selected_model)
            
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
st.sidebar.write(f"Stream Speed: {stream_speed}")

# Footer
st.sidebar.divider()
st.sidebar.markdown("Built with Streamlit + Ollama")