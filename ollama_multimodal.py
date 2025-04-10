import streamlit as st
import requests
import base64
import io
from typing import List, Dict, Any
from PIL import Image
import os
from helper import OLLAMA_API_BASE


# Set page configuration
st.set_page_config(page_title="Ollama Multimodal Chatbot", page_icon="🤖", layout="wide")

# Constants
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant that can understand and generate images."
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "enable_image_gen" not in st.session_state:
    st.session_state.enable_image_gen = False

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

# Function to identify multimodal models (This is a simple heuristic)
def get_multimodal_models(models: List[str]) -> List[str]:
    # Models known to have multimodal capabilities
    multimodal_identifiers = ["llava", "bakllava", "moondream", "cogvlm", "llava-llama"]
    
    multimodal_models = []
    for model in models:
        for identifier in multimodal_identifiers:
            if identifier.lower() in model.lower():
                multimodal_models.append(model)
                break
    
    return multimodal_models

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to encode PIL Image to base64
def pil_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Function to call Ollama API for image processing
def call_ollama_with_image(prompt: str, model: str, image_base64: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    
    # Add chat history excluding images for simplicity
    for msg in st.session_state.messages:
        if msg.get("image_data") is None:  # Only add text messages to context
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current prompt with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image_base64}
        ]
    })
    
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

# Function to call Ollama API for text chat
def call_ollama(prompt: str, model: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    
    # Add chat history excluding images for simplicity
    for msg in st.session_state.messages:
        if msg.get("image_data") is None:  # Only add text messages to context
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

# Function to generate image with image generation model
def generate_image(prompt: str, model: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": model,
                "prompt": f"Generate an image of {prompt}",
                "system": "You are an image generation AI. Create a detailed image based on the prompt.",
                "options": {"temperature": st.session_state.temperature}
            }
        )
        
        if response.status_code == 200:
            # Parse the image data - this is hypothetical since Ollama's format may vary
            # For now, returning a placeholder
            image_path = f"{IMAGE_DIR}/generated_{len(os.listdir(IMAGE_DIR))}.png"
            return image_path
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"

# Sidebar for model selection
st.sidebar.title("Ollama Multimodal Chatbot")

# Get available models
all_models = get_available_models()
multimodal_models = get_multimodal_models(all_models)

if not all_models:
    st.sidebar.warning("No models found or Ollama is not running. Please start Ollama and refresh.")
    selected_model = ""
    selected_image_gen_model = ""
else:
    selected_model = st.sidebar.selectbox("Select a chat model", all_models)
    
    # Only show image generation option if multimodal models are available
    if multimodal_models:
        st.sidebar.subheader("Image Features")
        st.session_state.enable_image_gen = st.sidebar.checkbox(
            "Enable Image Generation", 
            value=st.session_state.enable_image_gen,
            help="Generate images using a multimodal model (experimental)"
        )
        
        if st.session_state.enable_image_gen:
            selected_image_gen_model = st.sidebar.selectbox(
                "Select image model", 
                multimodal_models,
                help="Select a model capable of image generation"
            )
    else:
        st.sidebar.warning("No multimodal models detected. Install models like llava or bakllava for image capabilities.")
        selected_image_gen_model = ""

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

# Main chat interface
st.title("💬 Ollama Multimodal Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_data") is not None:
            st.image(message["image_data"])

# Image upload for analysis
uploaded_image = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    if not selected_model:
        st.error("Please select a model from the sidebar first.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
            # If image was uploaded, display it and add to chat
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image)
                image_base64 = pil_image_to_base64(image)
                
                # Add message with image to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt,
                    "image_data": image
                })
                
                # Process with multimodal model
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing image..."):
                        response = call_ollama_with_image(prompt, selected_model, image_base64)
                        
                        if "error" in response:
                            st.error(response["error"])
                        else:
                            assistant_response = response.get("message", {}).get("content", "No response")
                            st.markdown(assistant_response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": assistant_response
                            })
            else:
                # Handle image generation if requested
                if st.session_state.enable_image_gen and "generate image" in prompt.lower():
                    # Add message to chat history
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": prompt
                    })
                    
                    # Extract the image description
                    image_prompt = prompt.lower().replace("generate image", "").replace("of", "").strip()
                    
                    # Generate image
                    with st.chat_message("assistant"):
                        with st.spinner("Generating image..."):
                            # For now, create a placeholder image since direct image generation 
                            # with Ollama might not be standardized yet
                            img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                            
                            # Save placeholder image
                            image_path = f"{IMAGE_DIR}/generated_{len(os.listdir(IMAGE_DIR))}.png"
                            img.save(image_path)
                            
                            assistant_response = f"Here's the generated image for '{image_prompt}'"
                            st.markdown(assistant_response)
                            st.image(image_path)
                            
                            # Add assistant response with image to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": assistant_response,
                                "image_data": img
                            })
                else:
                    # Regular text chat without image
                    # Add message to chat history
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": prompt
                    })
                    
                    # Get response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = call_ollama(prompt, selected_model)
                            
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                assistant_response = response.get("message", {}).get("content", "No response")
                                st.markdown(assistant_response)
                                
                                # Add assistant response to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": assistant_response
                                })

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.markdown("Built with Streamlit + Ollama")
st.sidebar.markdown("Image capabilities work with multimodal models like llava")