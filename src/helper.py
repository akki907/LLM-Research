
import requests
import streamlit as st
from typing import List

OLLAMA_API_BASE = "http://localhost:11434/api"


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