import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List
import pytesseract
from PIL import Image
import io

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=API_KEY,
    task_type="retrieval_query" 
)

def extract_text_from_image(image):
    """Extract text from image using pytesseract"""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image = Image.open(io.BytesIO(image_bytes))
                img_text = extract_text_from_image(image)
                if img_text.strip():
                    text += f"\n[Image content: {img_text}]\n"
            except Exception as e:
                print(f"Error processing image: {str(e)}")
        
        all_text.append(text)
    
    return "\n\n".join(all_text)

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    if not chunks:
        print("Warning: No chunks to embed")
        return []
        
    try:
        embeddings = embedding_model.embed_documents(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            print(f"Warning: Expected {len(chunks)} embeddings, got {len(embeddings) if embeddings else 0}")
        return embeddings
    except Exception as e:
        print(f"Error embedding chunks: {str(e)}")
        raise

def view_pdf_with_unstructured(file_path: str):
    """
    Extract structured elements from PDF using PyMuPDF
    """
    doc = fitz.open(file_path)
    structured_elements = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for b_num, block in enumerate(blocks):
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
    
                            if "text" in span and span["text"].strip():
                                element_type = "Text"
                                if "size" in span and span["size"] > 12:
                                    element_type = "Heading"
                                    
                                structured_elements.append({
                                    "type": element_type,
                                    "text": span["text"],
                                    "metadata": {
                                        "page_number": page_num + 1,
                                        "font_size": span.get("size", "unknown"),
                                        "font": span.get("font", "unknown")
                                    }
                                })
            
            if block.get("type") == 1:  
                structured_elements.append({
                    "type": "Image",
                    "text": f"[Image on page {page_num + 1}]",
                    "metadata": {
                        "page_number": page_num + 1,
                        "width": block.get("width", 0),
                        "height": block.get("height", 0)
                    }
                })
        
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image = Image.open(io.BytesIO(image_bytes))
                img_text = extract_text_from_image(image)
                
                structured_elements.append({
                    "type": "Image",
                    "text": f"[Image on page {page_num + 1}]" + (f": {img_text}" if img_text.strip() else ""),
                    "metadata": {
                        "page_number": page_num + 1,
                        "ocr_text": img_text if img_text.strip() else None
                    }
                })
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                
    return structured_elements

def answer_question(question: str, context_docs: List[str]) -> str:
    """
    Use the Gemini API to answer a question based on the provided context
    """
    context = "\n\n".join(context_docs)
    
    prompt = f"""Please provide a comprehensive answer to the question.
    
Question: {question}

Context:
{context}

Instructions:
1.You are a helpful assistant.
2.Provide a detailed answer based on the context provided.
3.Use the context to support your answer.

Answer:"""

    try:

        model = genai.GenerativeModel('gemini-2.0-flash')
        
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 800,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        print(f"Error with gemini-2.0-flash: {str(e)}")
        
        try:
            model = genai.GenerativeModel('gemini-2.0-exp')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            print(f"Error with gemini-pro fallback: {str(e2)}")
            return f"I couldn't process this question due to an API issue: {str(e)}"
