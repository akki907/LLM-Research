.PHONY: install run clean

# Install dependencies
install:
	pip install -r requirements.txt

activate:
	pyhton3 -m venv venv   
	source venv/bin/activate 

# Run chatbot Streamlit app
run-ollama_chatbot:
	streamlit run src/ollama_chatbot.py

run-rag:
	streamlit run src/rag_chatbot.py

run-ollama_chatbot_stream:
	streamlit run src/ollma_chatbot_steam.py

run-multimodal:
	streamlit run src/ollama_multimodal.py

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +