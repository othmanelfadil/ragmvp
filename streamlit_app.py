import streamlit as st
import requests
import os
import time
import json
import asyncio
import httpx
from datetime import datetime
import pandas as pd
import plotly.express as px
from exam_generator import get_available_documents, generate_exam, generate_exercise
from ingestion import extract_text_from_pdf, chunk_text, embed_chunks
from chroma_store import store_chunks
from pdf_generator import generate_exam_pdf, generate_exercise_pdf
import base64
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000"  
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "chat"

if "exam_result" not in st.session_state:
    st.session_state.exam_result = None

if "exercise_result" not in st.session_state:
    st.session_state.exercise_result = None

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = get_available_documents()


def load_files():
    """Load the list of available files from the files directory"""
    try:
        files = os.listdir("files")
        pdf_files = [f for f in files if f.endswith('.pdf')]
        return pdf_files
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return []


async def upload_file_async(file):
    """Upload a file to the backend API asynchronously"""
    try:
        files = {"file": file}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_URL}/upload/", files=files)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                return {"error": f"Error uploading file: {response.text}"}
    except Exception as e:
        return {"error": f"Error uploading file: {str(e)}"}


def upload_file(file):
    """Upload a file to the backend API (synchronous wrapper)"""
    try:
        files = {"file": file}
        with st.spinner("Uploading and processing file..."):
            response = requests.post(f"{API_URL}/upload/", files=files)
            if response.status_code == 200:
                data = response.json()
                st.session_state.files = load_files()
                st.session_state.ingested_files = get_available_documents()
                return data
            else:
                st.error(f"Error uploading file: {response.text}")
                return None
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None


async def ingest_local_document_async(filename):
    """Ingest a document from the local files directory into ChromaDB asynchronously"""
    try:
        file_path = os.path.join("files", filename)
        if not os.path.exists(file_path):
            return False, f"File {filename} not found in files directory"
        
        
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(executor, extract_text_from_pdf, file_path)
        
       
        chunks = await loop.run_in_executor(executor, chunk_text, raw_text)
        
        
        embeddings = await loop.run_in_executor(executor, embed_chunks, chunks)
        
        
        chunks_stored = await loop.run_in_executor(executor, store_chunks, chunks, embeddings, filename)
        
        return True, f"Successfully ingested {chunks_stored} chunks from {filename}"
    except Exception as e:
        return False, f"Error ingesting document: {str(e)}"


def ingest_local_document(filename):
    """Ingest a document from the local files directory into ChromaDB (synchronous wrapper)"""
    try:
        file_path = os.path.join("files", filename)
        if not os.path.exists(file_path):
            return False, f"File {filename} not found in files directory"
        
        with st.spinner(f"Ingesting {filename} into vector database..."):
            raw_text = extract_text_from_pdf(file_path)
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            chunks_stored = store_chunks(chunks, embeddings, filename)
            st.session_state.ingested_files = get_available_documents()
            return True, f"Successfully ingested {chunks_stored} chunks from {filename}"
    except Exception as e:
        return False, f"Error ingesting document: {str(e)}"


async def ask_question_async(question, filename=None, num_results=5, recency_boost=True, content_length_weight=0.1):
    """Send a question to the backend API asynchronously"""
    try:
        data = {
            "question": question,
            "filename": filename,
            "num_results": num_results,
            "recency_boost": recency_boost,
            "content_length_weight": content_length_weight
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_URL}/qa/", json=data)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                data["response_time"] = round(response_time, 2)
                return data
            else:
                return {"error": f"Error getting answer: {response.text}"}
    except Exception as e:
        return {"error": f"Error getting answer: {str(e)}"}


def ask_question(question, filename=None, num_results=5, recency_boost=True, content_length_weight=0.1):
    """Send a question to the backend API (synchronous wrapper)"""
    try:
        data = {
            "question": question,
            "filename": filename,
            "num_results": num_results,
            "recency_boost": recency_boost,
            "content_length_weight": content_length_weight
        }
        
        start_time = time.time()
        
        response = requests.post(f"{API_URL}/qa/", json=data)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            data["response_time"] = round(response_time, 2)
            return data
        else:
            st.error(f"Error getting answer: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        return None


async def display_file_info_async(filename):
    """Display information about the selected file asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/chunks/?filename={filename}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Error getting file info: {response.text}"}
    except Exception as e:
        return {"error": f"Error getting file info: {str(e)}"}


def display_file_info(filename):
    """Display information about the selected file (synchronous wrapper)"""
    try:
        response = requests.get(f"{API_URL}/chunks/?filename={filename}")
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Error getting file info: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting file info: {e}")
        return None


def get_pdf_download_link(pdf_path, filename):
    """Generate a download link for a PDF file"""
    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download {filename}</a>'



st.sidebar.title("📚 RAG Chat")
st.sidebar.divider()

tab_options = ["Chat", "Generate Exam", "Generate Exercise"]
selected_tab = st.sidebar.radio("Navigation", tab_options)
st.session_state.current_tab = selected_tab.lower().replace(" ", "_")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    result = upload_file(uploaded_file)
    if result:
        st.sidebar.success(f"File uploaded: {result['filename']}")
        st.session_state.current_file = result['filename']


st.sidebar.divider()
st.sidebar.subheader("Available Files")
st.session_state.files = load_files()

if st.session_state.files:
    selected_file = st.sidebar.selectbox(
        "Select a file",
        options=["All Files"] + st.session_state.files,
        index=0,
        key="file_selector"
    )
    
    if selected_file != "All Files":
        st.session_state.current_file = selected_file
    else:
        st.session_state.current_file = None
        
    if st.session_state.current_file and st.sidebar.button("Show File Info"):
        file_info = display_file_info(st.session_state.current_file)
        if file_info:
            st.sidebar.write(f"Total chunks: {file_info['total']}")
            if file_info["chunks"]:
                sample_df = pd.DataFrame([{
                    "Chunk": i+1, 
                    "Length": len(chunk["text"])
                } for i, chunk in enumerate(file_info["chunks"][:10])])
                
                st.sidebar.dataframe(sample_df, hide_index=True)
else:
    st.sidebar.info("No files available. Please upload a PDF file.")


if st.session_state.current_tab == "chat":
    st.sidebar.divider()
    with st.sidebar.expander("Search Options"):
        num_results = st.slider("Context chunks", min_value=1, max_value=10, value=5)
        recency_boost = st.checkbox("Boost recent documents", value=True)
        content_weight = st.slider("Content length weight", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

    st.title("📚 PDF RAG Chat")
    scope_text = f"Currently chatting with: {st.session_state.current_file or 'All files'}"
    st.write(scope_text)


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            
            if message["role"] == "assistant" and "context" in message:
                with st.expander("View sources and context"):
                    st.write(f"🔍 Sources: {', '.join(message['sources'])}")
                    st.write(f"⏱️ Response time: {message['response_time']} seconds")
                    
                    
                    if message["context_snippets"]:
                        context_df = pd.DataFrame(message["context_snippets"])
                        st.dataframe(context_df, hide_index=True)


    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("_Thinking..._")
            
           
            response_container = st.container()
            
            
            if "asyncio" not in st.session_state:
                st.session_state.asyncio = {}
                
            
            async def get_response():
                response = await ask_question_async(
                    prompt, 
                    filename=st.session_state.current_file,
                    num_results=num_results,
                    recency_boost=recency_boost,
                    content_length_weight=content_weight
                )
                return response
            
            import nest_asyncio
            nest_asyncio.apply()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(get_response())
            
            if response and "error" not in response:
                thinking_placeholder.empty()
                with response_container:
                    st.write(response["answer"])
                
                assistant_message = {
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"],
                    "response_time": response.get("response_time", 0),
                    "context": response.get("context_used", {}),
                    "context_snippets": response.get("context_used", {}).get("snippets", [])
                }
                st.session_state.messages.append(assistant_message)
                
                with st.expander("View sources and context"):
                    st.write(f"🔍 Sources: {', '.join(response['sources'])}")
                    st.write(f"⏱️ Response time: {response.get('response_time', 0)} seconds")
                    
                    
                    if response.get("context_used", {}).get("snippets"):
                        context_df = pd.DataFrame(response["context_used"]["snippets"])
                        st.dataframe(context_df, hide_index=True)
            else:
                error_msg = response.get("error", "I couldn't process your question. Please try again.") if response else "Failed to get a response"
                thinking_placeholder.markdown(f"_Error: {error_msg}_")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

elif st.session_state.current_tab == "generate_exam":
    st.title("📝 Exam Generator")
    st.write("Generate custom exams based on your documents")
    
    st.session_state.ingested_files = get_available_documents()
    
    vector_docs = st.session_state.ingested_files
    all_docs = st.session_state.files
    
    st.subheader("Document Status")
    
    if not all_docs:
        st.warning("No documents available. Please upload a PDF file first.")
    else:
        status_data = []
        for doc in all_docs:
            ingested = doc in vector_docs
            status_data.append({
                "Document": doc,
                "Status": "Ingested ✅" if ingested else "Not Ingested ❌",
                "Ready for Exam": "Yes" if ingested else "No"
            })
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, hide_index=True)
        
        st.subheader("Prepare Document")
        non_ingested = [doc for doc in all_docs if doc not in vector_docs]
        
        if non_ingested:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                doc_to_ingest = st.selectbox(
                    "Select a document to ingest",
                    options=non_ingested,
                    key="doc_to_ingest"
                )
            
            with col2:
                if st.button("Ingest Document"):
                    status_placeholder = st.empty()
                    status_placeholder.info("Ingestion in progress...")
                    
                   
                   
                    import nest_asyncio
                    nest_asyncio.apply()
                    
                    async def run_ingestion():
                        success, message = await ingest_local_document_async(doc_to_ingest)
                        if success:
                            
                            st.session_state.ingested_files = get_available_documents()
                            status_placeholder.success(message)
                        else:
                            status_placeholder.error(message)
                        return success, message
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success, _ = loop.run_until_complete(run_ingestion())
                    
                    if success:
                        
                        st.rerun()
        else:
            st.success("All documents are ingested and ready for exam generation!")
    
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ingested_docs = [doc for doc in all_docs if doc in vector_docs]
            
            if not ingested_docs:
                st.warning("No ingested documents available for exam generation. Please ingest a document first.")
                exam_doc = None
            else:
                exam_doc = st.selectbox(
                    "Select document for exam",
                    options=ingested_docs,
                    key="exam_doc_selector"
                )
            
            if exam_doc:
                difficulty = st.select_slider(
                    "Difficulty level",
                    options=["easy", "medium", "hard"],
                    value="medium"
                )
                
                exam_type = st.selectbox(
                    "Exam type",
                    options=["mixed", "mcq", "essay", "fill_blank", "true_false"],
                    index=0
                )
                
                num_questions = st.slider(
                    "Number of questions",
                    min_value=3,
                    max_value=15,
                    value=5
                )
                
                if st.button("Generate Exam"):
                    with st.spinner("Generating exam questions..."):
                        result = generate_exam(
                            document_name=exam_doc,
                            difficulty=difficulty,
                            exam_type=exam_type,
                            num_questions=num_questions
                        )
                        
                        if result and result.get("success", False):
                            st.session_state.exam_result = result
                            st.success("Exam generated successfully!")
                        else:
                            st.error(f"Failed to generate exam: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.session_state.exam_result:
                exam = st.session_state.exam_result
                st.markdown(f"### {exam.get('title', 'Generated Exam')}")
                st.markdown(f"*{exam.get('description', '')}*")
                st.markdown(f"**Difficulty:** {exam.get('difficulty', 'medium')}")
                st.markdown(f"**Source:** {exam.get('source_document', '')}")
                
                questions = exam.get("questions", [])
                if questions:
                    for i, q in enumerate(questions):
                        with st.expander(f"Question {i+1}: {q.get('question', '')[:50]}..."):
                            st.markdown(f"**{q.get('question', '')}**")
                            st.markdown(f"*Type: {q.get('type', '')}*")
                            
                            if q.get('type') in ['mcq', 'true_false'] and 'options' in q:
                                for j, opt in enumerate(q.get('options', [])):
                                    st.markdown(f"{chr(65+j)}. {opt}")
                            
                            if st.toggle(f"Show answer for question {i+1}", key=f"toggle_q_{i}"):
                                st.markdown("**Answer:**")
                                st.markdown(q.get('answer', 'No answer provided'))
                                if 'explanation' in q:
                                    st.markdown("**Explanation:**")
                                    st.markdown(q.get('explanation', ''))
                
                st.divider()
                st.subheader("Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Export Exam (JSON)",
                        data=json.dumps(exam, indent=2),
                        file_name=f"exam_{exam.get('source_document', 'document')}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    include_answers = st.checkbox("Include answers in PDF", value=False)
                    
                    if st.button("Generate PDF"):
                        with st.spinner("Generating PDF..."):
                            pdf_path = generate_exam_pdf(exam, include_answers=include_answers)
                            
                            if pdf_path:
                                pdf_filename = f"exam_{exam.get('source_document', 'document')}_{datetime.now().strftime('%Y%m%d')}"
                                if include_answers:
                                    pdf_filename += "_with_answers"
                                pdf_filename += ".pdf"
                                
                                st.markdown(
                                    get_pdf_download_link(pdf_path, pdf_filename),
                                    unsafe_allow_html=True
                                )
                                st.success(f"PDF generated successfully! Click the link above to download.")
                            else:
                                st.error("Failed to generate PDF")
            else:
                st.info("Select a document and generate an exam to see results here.")

elif st.session_state.current_tab == "generate_exercise":
    st.title("📋 Exercise Generator")
    st.write("Create learning exercises and activities based on your documents")
    
    st.session_state.ingested_files = get_available_documents()
    
    vector_docs = st.session_state.ingested_files
    all_docs = st.session_state.files
    
    st.subheader("Document Status")
    
    if not all_docs:
        st.warning("No documents available. Please upload a PDF file first.")
    else:
        status_data = []
        for doc in all_docs:
            ingested = doc in vector_docs
            status_data.append({
                "Document": doc,
                "Status": "Ingested ✅" if ingested else "Not Ingested ❌",
                "Ready for Exercise": "Yes" if ingested else "No"
            })
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, hide_index=True)
        
        st.subheader("Prepare Document")
        non_ingested = [doc for doc in all_docs if doc not in vector_docs]
        
        if non_ingested:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                doc_to_ingest = st.selectbox(
                    "Select a document to ingest",
                    options=non_ingested,
                    key="exercise_doc_to_ingest"
                )
            
            with col2:
                if st.button("Ingest Document", key="ingest_for_exercise"):
                    status_placeholder = st.empty()
                    status_placeholder.info("Ingestion in progress...")
                    
                
                    import nest_asyncio
                    nest_asyncio.apply()
                    
                    async def run_ingestion():
                        success, message = await ingest_local_document_async(doc_to_ingest)
                        if success:
                            st.session_state.ingested_files = get_available_documents()
                            status_placeholder.success(message)
                        else:
                            status_placeholder.error(message)
                        return success, message
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success, _ = loop.run_until_complete(run_ingestion())
                    
                    if success:
                        st.rerun()
        else:
            st.success("All documents are ingested and ready for exercise generation!")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ingested_docs = [doc for doc in all_docs if doc in vector_docs]
            
            if not ingested_docs:
                st.warning("No ingested documents available for exercise generation. Please ingest a document first.")
                exercise_doc = None
            else:
                exercise_doc = st.selectbox(
                    "Select document for exercise",
                    options=ingested_docs,
                    key="exercise_doc_selector"
                )
            
            if exercise_doc:
                exercise_type = st.selectbox(
                    "Exercise type",
                    options=["practice", "worksheet", "challenge"],
                    index=0
                )
                
                difficulty = st.select_slider(
                    "Difficulty level",
                    options=["easy", "medium", "hard"],
                    value="medium"
                )
                
                focus_area = st.text_input(
                    "Focus area (optional)",
                    placeholder="Enter a specific topic or concept to focus on"
                )
                
                if st.button("Generate Exercise"):
                    with st.spinner("Creating exercise..."):
                        result = generate_exercise(
                            document_name=exercise_doc,
                            exercise_type=exercise_type,
                            difficulty=difficulty,
                            focus_area=focus_area if focus_area else None
                        )
                        
                        if result and result.get("success", False):
                            st.session_state.exercise_result = result
                            st.success("Exercise generated successfully!")
                        else:
                            st.error(f"Failed to generate exercise: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.session_state.exercise_result:
                exercise = st.session_state.exercise_result
                st.markdown(f"### {exercise.get('title', 'Generated Exercise')}")
                st.markdown(f"*{exercise.get('description', '')}*")
                st.markdown(f"**Difficulty:** {exercise.get('difficulty', 'medium')}")
                st.markdown(f"**Estimated time:** {exercise.get('estimated_time', '30 minutes')}")
                st.markdown(f"**Source:** {exercise.get('source_document', '')}")
                
                components = exercise.get("components", [])
                if components:
                    for i, comp in enumerate(components):
                        with st.expander(f"{i+1}. {comp.get('title', f'Component {i+1}')}"):
                            st.markdown(f"**Type:** {comp.get('type', '')}")
                            st.markdown(comp.get('content', ''))
                            
                            if 'instructions' in comp and comp['instructions']:
                                st.markdown("**Instructions:**")
                                st.markdown(comp['instructions'])
                
                with st.expander("Solution Guide"):
                    st.markdown(exercise.get('solution_guide', 'No solution guide provided'))
                
                st.divider()
                st.subheader("Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Export Exercise (JSON)",
                        data=json.dumps(exercise, indent=2),
                        file_name=f"exercise_{exercise.get('source_document', 'document')}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    include_solutions = st.checkbox("Include solutions in PDF", value=False)
                    
                    if st.button("Generate PDF"):
                        with st.spinner("Generating PDF..."):
                            pdf_path = generate_exercise_pdf(exercise, include_solutions=include_solutions)
                            
                            if pdf_path:
                                pdf_filename = f"exercise_{exercise.get('source_document', 'document')}_{datetime.now().strftime('%Y%m%d')}"
                                if include_solutions:
                                    pdf_filename += "_with_solutions"
                                pdf_filename += ".pdf"
                                
                                st.markdown(
                                    get_pdf_download_link(pdf_path, pdf_filename),
                                    unsafe_allow_html=True
                                )
                                st.success(f"PDF generated successfully! Click the link above to download.")
                            else:
                                st.error("Failed to generate PDF")
            else:
                st.info("Select a document and generate an exercise to see results here.")

if st.session_state.current_tab == "chat" and (st.session_state.exam_result or st.session_state.exercise_result):
    st.session_state.exam_result = None
    st.session_state.exercise_result = None