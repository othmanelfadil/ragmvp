from chromadb import Client
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import os

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

chroma_client = Client(Settings(
    persist_directory="./chroma",
    anonymized_telemetry=False
))

collection = chroma_client.get_or_create_collection(
    name="ingested_docs",
    metadata={"hnsw:space": "cosine"}  #cosine distance for better semantic matching
)

def get_collection():
    """Return the ChromaDB collection object"""
    return collection

def store_chunks(chunks, embeddings, filename):
    """Store chunks in ChromaDB with proper error handling"""
    if not chunks or not embeddings:
        print(f"Warning: No chunks or embeddings to store for {filename}")
        return 0
        
    if len(chunks) != len(embeddings):
        print(f"Error: Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
        return 0
        
    try:
        existing_ids = collection.get(
            where={"filename": filename},
            include=[]
        )["ids"]
        
        if existing_ids:
            print(f"Deleting {len(existing_ids)} existing chunks for {filename}")
            collection.delete(ids=existing_ids)
        
       
        doc_ids = [f"{filename}-{i}" for i in range(len(chunks))]
        
        metadatas = [{"filename": filename} for _ in chunks]
        
        collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        if DEBUG:
            print(f"Successfully stored {len(chunks)} chunks for file '{filename}'")
            
        count = len(collection.get(
            where={"filename": filename},
            include=[]
        )["ids"])
        
        print(f"Verified {count} chunks stored for {filename}")
        return count
        
    except Exception as e:
        print(f"Error storing chunks: {str(e)}")
        raise

def custom_search(
    query_embedding: List[float], 
    k: int = 5, 
    filename_filter: str = None,
    recency_boost: bool = True,
    content_length_weight: float = 0.1
) -> Dict[str, Any]:
    """
    Custom search algorithm that combines vector similarity with additional weighting factors
    """
    all_ids = collection.get(include=[])["ids"]
    if len(all_ids) == 0:
        if DEBUG:
            print("Warning: Collection is empty, no documents to search")
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
    
    search_k = min(k * 4, 30)           
    where_filter = {"filename": filename_filter} if filename_filter else None
    

    try:
        if DEBUG:
            print(f"Searching for content with {'filename filter: ' + filename_filter if filename_filter else 'no filename filter'}")
        

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k,
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        if (not results["documents"] or not results["documents"][0]) and filename_filter:
            if DEBUG:
                print("No results with filter, trying without filter")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
    
    except Exception as e:
        if DEBUG:
            print(f"Error in vector search: {str(e)}")
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
    
    if not results["documents"] or not results["documents"][0]:
        if DEBUG:
            print("No results found in vector search")
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
    
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    doc_ids_numeric = [int(doc_id.split('-')[-1]) for doc_id in ids]
    max_id = max(doc_ids_numeric) if doc_ids_numeric else 0
    
    scores = []
    for i, (distance, doc, doc_id_num) in enumerate(zip(distances, documents, doc_ids_numeric)):
        similarity = 1 - min(distance, 1.0)
        
        length_score = min(len(doc) / 1000, 1.0) * content_length_weight
        
        recency_score = 0
        if recency_boost and max_id > 0:
            recency_score = (doc_id_num / max_id) * 0.1
        
        final_score = similarity + length_score + recency_score
        scores.append(final_score)
    
    reranked_indices = np.argsort(scores)[::-1][:k]
    
    reranked_results = {
        "ids": [ids[i] for i in reranked_indices],
        "documents": [documents[i] for i in reranked_indices],
        "metadatas": [metadatas[i] for i in reranked_indices],
        "distances": [1 - scores[i] for i in reranked_indices],
        "custom_scores": [scores[i] for i in reranked_indices]
    }
    
    result_format = {
        "ids": [reranked_results["ids"]],
        "documents": [reranked_results["documents"]],
        "metadatas": [reranked_results["metadatas"]],
        "distances": [reranked_results["distances"]],
        "custom_scores": [reranked_results["custom_scores"]]
    }
    
    if DEBUG and result_format["documents"][0]:
        print(f"Found {len(result_format['documents'][0])} results after reranking")
    
    return result_format
