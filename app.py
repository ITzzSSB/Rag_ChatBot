
import streamlit as st
import requests
from typing import List
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

# Configure page
st.set_page_config(
    page_title="Document QA with Gemini",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stFileUploader>div>div>button {
        background-color: #4285F4;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f1f1f1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Helper functions
def call_backend(endpoint: str, method: str = "get", data=None, files=None):
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method.lower() == "get":
            response = requests.get(url)
        elif method.lower() == "post":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def upload_file(file):
    if file is not None:
        with st.spinner(f"Processing {file.name}..."):
            response = call_backend(
                "/upload/",
                method="post",
                files={"file": (file.name, file.getvalue(), file.type)}
            )
            if response and response.get("status") == "success":
                st.success("File processed successfully!")
                st.session_state.processed_files.append(file.name)
                with st.expander("File Preview"):
                    st.text(response.get("preview", "No preview available"))
                return True
    return False

def ask_question(question: str):
    if not question:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.spinner("Thinking..."):
        response = call_backend(
            "/ask/",
            method="post",
            data={"question": question}
        )
        
        if response and response.get("status") == "success":
            answer = response.get("answer", "No answer found")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't process your question."})

def reset_session():
    response = call_backend("/reset/", method="post")
    if response and response.get("status") == "success":
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.sidebar.success("Session reset successfully!")
    else:
        st.sidebar.error("Failed to reset session")

# Sidebar for file upload
st.sidebar.title("Document Management")
st.sidebar.markdown("Upload your documents for analysis")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file (PDF)",
    type=["pdf"],
    key="file_uploader"
)

if uploaded_file and upload_file(uploaded_file):
    uploaded_file = None  # Reset uploader after successful upload

# Display processed files
if st.session_state.processed_files:
    st.sidebar.subheader("Processed Files")
    for file in st.session_state.processed_files:
        st.sidebar.markdown(f"- {file}")

# Check backend connection
with st.sidebar:
    if st.button("Check Backend Connection"):
        response = call_backend("/health")
        if response:
            st.success(f"Backend connected! Model: {response.get('model', 'Unknown')}")
        else:
            st.error("Backend connection failed")

# Reset button
st.sidebar.button("Reset Session", on_click=reset_session)

# Main chat interface
st.title("📄 Document QA with Gemini")
st.markdown("Ask questions about your uploaded documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    ask_question(prompt)
    st.rerun()

# Instructions expander
with st.expander("How to use this app"):
    st.markdown("""
    1. **Upload documents** using the sidebar (PDF)
    2. **Ask questions** about the content in the chat box below
    3. The AI will answer based on the documents you've uploaded
    
    **Features:**
    - Supports multiple document types
    - Maintains conversation history
    - Shows document previews
    - Powered by Google Gemini
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**Environment Setup**")
