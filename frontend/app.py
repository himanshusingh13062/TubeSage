import streamlit as st
import requests
import json
from typing import Dict, List
import time
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def load_css(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")
        st.info("Make sure 'styles.css' is in the same directory as your app.py")

load_css('styles.css')

BACKEND_URL = "http://localhost:8000"
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_setup" not in st.session_state:
    st.session_state.rag_setup = False

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/")
        return response.status_code == 200
    except:
        return False

def get_system_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def setup_rag_youtube(video_id: str):
    try:
        response = requests.post(f"{BACKEND_URL}/setup/youtube", json={
            "video_id": video_id,
            "google_api_key": GOOGLE_API_KEY
        })
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def setup_rag_text(text: str):
    try:
        response = requests.post(f"{BACKEND_URL}/setup/text", json={
            "text": text,
            "google_api_key": GOOGLE_API_KEY
        })
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def query_rag(question: str):
    try:
        response = requests.post(f"{BACKEND_URL}/query", json={
            "question": question
        })
        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def clear_memory():
    try:
        response = requests.delete(f"{BACKEND_URL}/memory")
        return response.status_code == 200
    except:
        return False

def extract_video_id(url_or_id: str) -> str:
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        if "watch?v=" in url_or_id:
            return url_or_id.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url_or_id:
            return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id

# Main UI
st.markdown('<h1 class="main-header">YouTube RAG Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with YouTube videos using AI</p>', unsafe_allow_html=True)

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please add GOOGLE_API_KEY to your .env file")
    st.info("Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
    st.stop()

if not check_backend_status():
    st.error("Backend server is not running. Please start it with: uvicorn main:app --reload")
    st.stop()

status = get_system_status()
if status:
    if status["rag_initialized"] and status["has_vectorstore"]:
        st.markdown('<div class="status-good">System Ready - Chat with your content!</div>', unsafe_allow_html=True)
        st.session_state.rag_setup = True
    else:
        st.markdown('<div class="status-warning">Please setup your knowledge base first</div>', unsafe_allow_html=True)

if not st.session_state.rag_setup:
    st.markdown('<div class="setup-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="setup-header">Setup Knowledge Base</h3>', unsafe_allow_html=True)
    
    setup_option = st.radio(
        "Choose your content source:",
        ["YouTube Video", "Custom Text"],
        horizontal=True
    )
    
    if setup_option == "YouTube Video":
        st.markdown("#### Enter YouTube Video")
        video_input = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://www.youtube.com/watch?v=xvFZjo5PgG0 or xvFZjo5PgG0 ",
            help="Paste a YouTube URL or just the video ID"
        )
        
        if st.button("Setup with YouTube", type="primary"):
            if video_input.strip():
                video_id = extract_video_id(video_input.strip())
                with st.spinner("Processing YouTube video..."):
                    success, result = setup_rag_youtube(video_id)
                    if success:
                        st.success("YouTube video processed successfully!")
                        st.session_state.rag_setup = True
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Setup failed: {result.get('detail', 'Unknown error')}")
            else:
                st.warning("Please provide a YouTube URL or video ID")
    
    else:  
        st.markdown("#### Enter Your Text")
        text_input = st.text_area(
            "Custom Content",
            placeholder="Paste your text, article, or document here...",
            height=200,
            help="Add any text content you want to chat with"
        )
        
        if st.button("Setup with Text", type="primary"):
            if text_input.strip():
                with st.spinner("Processing your text..."):
                    success, result = setup_rag_text(text_input.strip())
                    if success:
                        st.success("Text processed successfully!")
                        st.session_state.rag_setup = True
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Setup failed: {result.get('detail', 'Unknown error')}")
            else:
                st.warning("Please provide some text content")
    
    st.markdown('</div>', unsafe_allow_html=True)


else:
    st.markdown('<div class="section-header">Chat</div>', unsafe_allow_html=True)
    
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="empty-chat">Start a conversation by asking a question below</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask anything about your content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            success, result = query_rag(prompt)
            
            if success:
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_msg = f"Error: {result.get('detail', 'Something went wrong')}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear Memory"):
            if clear_memory():
                st.success("Memory cleared!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to clear memory")
    
    with col3:
        if st.button("New Session"):
            st.session_state.rag_setup = False
            st.session_state.messages = []
            clear_memory()
            st.rerun()

st.markdown(
    '<div class="footer-text">Built by Himanshu Singh</div>',
    unsafe_allow_html=True
)