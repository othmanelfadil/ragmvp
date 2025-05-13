from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import os
import fitz 
import tempfile
from pydantic import BaseModel
from typing import Optional, List

from ingestion import extract_text_from_pdf, chunk_text as text_chunker, embed_chunks, view_pdf_with_unstructured, answer_question
from chroma_store import store_chunks, get_collection, custom_search

app = FastAPI()
UPLOAD_FOLDER = "files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_text = extract_text_from_pdf(file_path)
    chunks = text_chunker(raw_text)  
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, file.filename)

    sample_size = min(3, len(chunks))
    sample_chunks = chunks[:sample_size]

    return { 
        "status": "success",
        "message": f"PDF '{file.filename}' successfully embedded into ChromaDB vector store",
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "sample_chunks": sample_chunks
    }

class QuestionRequest(BaseModel):
    question: str
    filename: str = None
    num_results: int = 5
    recency_boost: bool = True
    content_length_weight: float = 0.1

@app.post("/qa/")
async def question_answering(request: QuestionRequest):
    """
    Answer questions using the Gemini API with custom semantic search algorithm
    """
    try:
        from ingestion import embedding_model
        question_embedding = embedding_model.embed_query(request.question)
        
        results = custom_search(
            query_embedding=question_embedding,
            k=request.num_results,
            filename_filter=request.filename,
            recency_boost=request.recency_boost,
            content_length_weight=request.content_length_weight
        )
        
        if not results.get("documents") or not results["documents"] or not results["documents"][0]:
            return {
                "question": request.question,
                "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing or ask a different question.",
                "error": "No relevant content found",
                "sources": [],
                "context_used": {"count": 0, "snippets": []}
            }
        
        context_docs = results["documents"][0]
        source_files = [meta["filename"] for meta in results["metadatas"][0]]
        
        custom_scores = results["custom_scores"][0] if "custom_scores" in results else None
        
        answer = answer_question(request.question, context_docs)
        
        context_snippets = []
        for i, (doc, meta) in enumerate(zip(context_docs, results["metadatas"][0])):
            snippet = {
                "content": doc[:150] + "..." if len(doc) > 150 else doc,
                "source": meta["filename"],
                "relevance_score": round(custom_scores[i], 2) if custom_scores else None
            }
            context_snippets.append(snippet)
        
        return {
            "question": request.question,
            "answer": answer,
            "sources": list(set(source_files)),
            "search_method": "Custom semantic search with content length and recency factors",
            "context_used": {
                "count": len(context_docs),
                "snippets": context_snippets
            },
            "search_scope": "Limited to " + request.filename if request.filename else "All documents (custom semantic search)"
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing question: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the error for debugging
        
        return {
            "question": request.question,
            "answer": "I encountered an error while trying to answer your question.",
            "error": str(e),
            "error_detail": error_detail if os.getenv("DEBUG", "False").lower() == "true" else None
        }

@app.get("/chunks/")
async def get_all_chunks(
    filename: Optional[str] = None, 
    page: int = Query(1, ge=1), 
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get all chunks with optional filtering by filename and pagination
    """
    try:
        collection = get_collection()
        
        where_filter = {"filename": filename} if filename else None
        
        offset = (page - 1) * limit
        
        results = collection.get(
            where=where_filter,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        
        if filename:
            total_count = len(collection.get(where={"filename": filename}, include=[])["ids"])
        else:
            total_count = len(collection.get(include=[])["ids"])
        
        chunks = []
        for i in range(len(results["ids"])):
            embedding_preview = []
            if "embeddings" in results and i < len(results["embeddings"]) and results["embeddings"][i] is not None:
                embedding_preview = results["embeddings"][i][:5]
            
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "embedding_preview": [round(v, 4) for v in embedding_preview] + ["..."]
            })
        
        return {
            "total": total_count,
            "page": page,
            "limit": limit,
            "total_pages": (total_count + limit - 1) // limit,
            "chunks": chunks,
            "filter": {"filename": filename} if filename else "None"
        }
    except Exception as e:
        import traceback
        error_detail = f"Error retrieving chunks: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the error
        
        return {
            "error": str(e),
            "detail": error_detail if os.getenv("DEBUG", "False").lower() == "true" else "Error retrieving chunks"
        }





