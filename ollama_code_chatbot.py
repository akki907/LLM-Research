import streamlit as st
import requests
import json
from typing import List, Dict, Any
import re
import io
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import traceback
from io import StringIO
import ast
import time
import subprocess
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Ollama Code Execution Chatbot", page_icon="🤖", layout="wide")

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"
SUPPORTED_LANGUAGES = ["python", "bash", "javascript"]

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful coding assistant. When writing code, make sure to use markdown code blocks with the language specified (e.g. ```python)."
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "auto_execute_code" not in st.session_state:
    st.session_state.auto_execute_code = False
if "execution_history" not in st.session_state:
    st.session_state.execution_history = {}

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

# Function to extract code blocks from markdown text
def extract_code_blocks(text):
    # Pattern to match markdown code blocks
    pattern = r"```(\w+)?\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)
    code_blocks = []
    
    for lang, code in matches:
        # Default to python if language not specified
        language = lang.strip() if lang.strip() else "python"
        code_blocks.append({
            "language": language.lower(),
            "code": code
        })
    
    return code_blocks

# Function to check if code is safe to execute
def is_safe_code(code, language="python"):
    if language != "python":
        # For non-Python code, use a basic blocklist approach
        dangerous_commands = [
            "rm -rf", "format", "mkfs", "dd if=/dev/zero", 
            ":(){:|:&};:", "chmod -R 777", "> /dev/sda",
            "eval", "exec", "fork bomb", "wget", "curl"
        ]
        for cmd in dangerous_commands:
            if cmd in code.lower():
                return False, f"Potentially dangerous command detected: {cmd}"
        return True, "Code appears safe"
    
    # For Python, use AST parsing to detect potentially dangerous operations
    try:
        tree = ast.parse(code)
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in ['os', 'subprocess', 'sys', 'shutil']:
                        self.issues.append(f"Importing potentially dangerous module: {name.name}")
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                if node.module in ['os', 'subprocess', 'sys', 'shutil']:
                    self.issues.append(f"Importing from potentially dangerous module: {node.module}")
                self.generic_visit(node)
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'exec', 'eval', 'execfile', 'remove', 'rmdir', 'unlink']:
                        self.issues.append(f"Potentially dangerous function call: {node.func.attr}")
                elif isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'execfile']:
                        self.issues.append(f"Potentially dangerous function call: {node.func.id}")
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        if visitor.issues:
            return False, "Security issues found: " + ", ".join(visitor.issues)
        return True, "Code appears safe based on static analysis"
        
    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"

# Function to execute Python code in a sandbox
def execute_python_code(code):
    # Create string buffers to capture output
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # Create a dictionary for locals to capture variables
    local_vars = {}
    
    # Setup the execution environment
    execution_globals = {
        'np': np,
        'pd': pd,
        'plt': plt,
        'print': print,
        # Add other safe modules as needed
    }
    
    result = {
        'stdout': '',
        'stderr': '',
        'exception': None,
        'figures': [],
        'dataframes': [],
        'execution_time': 0,
        'return_value': None
    }
    
    # Execute the code with redirected stdout/stderr
    start_time = time.time()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute code and capture the last expression as a return value
            parsed = ast.parse(code)
            if parsed.body:
                last = parsed.body[-1]
                
                # If the last statement is an expression, capture its value
                if isinstance(last, ast.Expr):
                    parsed.body = parsed.body[:-1]
                    compiled_code = compile(parsed, '<string>', 'exec')
                    exec(compiled_code, execution_globals, local_vars)
                    
                    # Now evaluate the last expression
                    last_expr = compile(ast.Expression(last.value), '<string>', 'eval')
                    result['return_value'] = eval(last_expr, execution_globals, local_vars)
                else:
                    # No return value to capture, just execute everything
                    compiled_code = compile(parsed, '<string>', 'exec')
                    exec(compiled_code, execution_globals, local_vars)
    except Exception as e:
        result['exception'] = traceback.format_exc()
    
    end_time = time.time()
    result['execution_time'] = end_time - start_time
    
    # Capture output
    result['stdout'] = stdout_buffer.getvalue()
    result['stderr'] = stderr_buffer.getvalue()
    
    # Check if there were any matplotlib figures created
    if 'plt' in execution_globals and plt.get_fignums():
        for i in plt.get_fignums():
            fig = plt.figure(i)
            result['figures'].append(fig)
        plt.close('all')  # Close all figures
    
    # Check for DataFrames in local variables
    for var_name, var_value in local_vars.items():
        if isinstance(var_value, pd.DataFrame) and len(var_value) > 0:
            result['dataframes'].append((var_name, var_value))
    
    return result

# Function to execute bash commands safely
def execute_bash_code(code):
    result = {
        'stdout': '',
        'stderr': '',
        'exception': None,
        'execution_time': 0
    }
    
    # Create a temporary file to write the script
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name
    
    try:
        start_time = time.time()
        # Execute with restricted permissions and timeout
        process = subprocess.Popen(
            ['bash', temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Set a timeout for the process
        try:
            stdout, stderr = process.communicate(timeout=10)
            result['stdout'] = stdout
            result['stderr'] = stderr
            result['return_code'] = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            result['exception'] = "Execution timed out after 10 seconds"
        
        end_time = time.time()
        result['execution_time'] = end_time - start_time
        
    except Exception as e:
        result['exception'] = str(e)
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return result

# Function to execute JavaScript code using Node.js
def execute_javascript_code(code):
    result = {
        'stdout': '',
        'stderr': '',
        'exception': None,
        'execution_time': 0
    }
    
    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print('node is installed')
    except (subprocess.SubprocessError, FileNotFoundError):
        result['exception'] = "Node.js is not installed or not in PATH. Cannot execute JavaScript code."
        return result
    
    # Create a temporary file to write the script
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.js') as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name
    
    try:
        start_time = time.time()
        # Execute with timeout
        process = subprocess.Popen(
            ['node', temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Set a timeout for the process
        try:
            stdout, stderr = process.communicate(timeout=10)
            result['stdout'] = stdout
            result['stderr'] = stderr
            result['return_code'] = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            result['exception'] = "Execution timed out after 10 seconds"
        
        end_time = time.time()
        result['execution_time'] = end_time - start_time
        
    except Exception as e:
        result['exception'] = str(e)
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return result

# Function to execute code based on language
def execute_code(code_block):
    language = code_block["language"].lower()
    code = code_block["code"]
    
    # Check if code is safe to execute
    is_safe, safety_message = is_safe_code(code, language)
    if not is_safe:
        return {
            'stdout': '',
            'stderr': '',
            'exception': f"Code execution blocked: {safety_message}",
            'execution_time': 0
        }
    
    # Execute based on language
    if language == "python":
        return execute_python_code(code)
    elif language == "bash" or language == "shell":
        return execute_bash_code(code)
    elif language == "javascript" or language == "js":
        return execute_javascript_code(code)
    else:
        return {
            'stdout': '',
            'stderr': '',
            'exception': f"Execution not supported for language: {language}",
            'execution_time': 0
        }

# Function to call Ollama API
def call_ollama(prompt: str, model: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    
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

# Sidebar setup
st.sidebar.title("Ollama Code Execution Chatbot")

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

# Code execution settings
st.sidebar.subheader("Code Execution Settings")
st.session_state.auto_execute_code = st.sidebar.checkbox(
    "Auto-execute code", 
    value=st.session_state.auto_execute_code,
    help="Automatically run code blocks when received"
)

# Main chat interface
st.title("💬 Ollama Code Execution Chatbot")

# Display chat messages with code execution buttons
for i, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        # Check if we need to extract and potentially execute code blocks
        if role == "assistant":
            # Extract code blocks
            code_blocks = extract_code_blocks(content)
            
            # Display content with proper markdown
            st.markdown(content)
            
            # If there are code blocks, add execution buttons
            if code_blocks:
                for j, code_block in enumerate(code_blocks):
                    language = code_block["language"]
                    code = code_block["code"]
                    
                    # Only offer execution for supported languages
                    if language.lower() in SUPPORTED_LANGUAGES:
                        # Create a unique key for this code block
                        block_key = f"block_{i}_{j}"
                        
                        # Check if this block has already been executed
                        if block_key in st.session_state.execution_history:
                            # Show previous execution results
                            with st.expander(f"Execution Results (Time: {st.session_state.execution_history[block_key]['execution_time']:.2f}s)", expanded=True):
                                result = st.session_state.execution_history[block_key]
                                
                                if result['stdout']:
                                    st.subheader("Output:")
                                    st.code(result['stdout'])
                                
                                if result['stderr']:
                                    st.subheader("Errors:")
                                    st.code(result['stderr'], language="bash")
                                
                                if result['exception']:
                                    st.error(result['exception'])
                                
                                # Show return value if present
                                if 'return_value' in result and result['return_value'] is not None:
                                    st.subheader("Return Value:")
                                    st.write(result['return_value'])
                                
                                # Show figures if any were generated
                                if 'figures' in result and result['figures']:
                                    st.subheader("Figures:")
                                    for fig in result['figures']:
                                        st.pyplot(fig)
                                
                                # Show dataframes if any were created
                                if 'dataframes' in result and result['dataframes']:
                                    st.subheader("DataFrames:")
                                    for df_name, df in result['dataframes']:
                                        st.write(f"DataFrame: {df_name}")
                                        st.dataframe(df)
                        else:
                            # Add an execution button
                            if st.button(f"Run {language} code", key=f"run_{i}_{j}"):
                                result = execute_code(code_block)
                                
                                # Store the execution results
                                st.session_state.execution_history[block_key] = result
                                
                                # Force a rerun to display the results
                                st.rerun()
        else:
            # Just display user messages normally
            st.markdown(content)

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
                    
                    # Display the response
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
                    # Check for code blocks to potentially auto-execute
                    if st.session_state.auto_execute_code:
                        code_blocks = extract_code_blocks(assistant_response)
                        
                        # Auto-execute the first code block if it's a supported language
                        if code_blocks:
                            for j, code_block in enumerate(code_blocks):
                                language = code_block["language"].lower()
                                
                                if language in SUPPORTED_LANGUAGES:
                                    # Create a unique key for this code block
                                    block_key = f"block_{len(st.session_state.messages) - 1}_{j}"
                                    
                                    # Execute the code
                                    with st.spinner(f"Auto-executing {language} code..."):
                                        result = execute_code(code_block)
                                        
                                        # Store the execution results
                                        st.session_state.execution_history[block_key] = result
                        
                        # Force a rerun to show the execution results
                        st.rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.execution_history = {}
    st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.markdown("Built with Streamlit + Ollama")
st.sidebar.markdown(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")