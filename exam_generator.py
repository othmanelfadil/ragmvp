import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
from chroma_store import get_collection
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=API_KEY,
    task_type="retrieval_query" 
)

def get_document_content(filename: str) -> str:
    """Retrieve all content from a specific document in the vector store"""
    collection = get_collection()
    results = collection.get(
        where={"filename": filename},
        include=["documents"]
    )
    
    if not results or not results["documents"]:
        return ""
    
    return "\n\n".join(results["documents"])

def get_available_documents() -> List[str]:
    """Get list of all documents available in the vector store"""
    collection = get_collection()
    results = collection.get(
        include=["metadatas"]
    )
    
    if not results or not results["metadatas"]:
        return []
    
    unique_filenames = set()
    for metadata in results["metadatas"]:
        if "filename" in metadata:
            unique_filenames.add(metadata["filename"])
    
    return list(unique_filenames)

def detect_document_type(document_name: str, content: str) -> Dict[str, Any]:
    """
    Detect document type and subject to adjust generation parameters
    
    Returns dict with:
    - type: "history", "science", "math", "literature", "economics", "physics", etc.
    - language: "english", "french", etc.
    - academic_level: "primary", "secondary", "undergraduate", "graduate", "professional"
    - specialized_areas: list of specialized topics detected
    """
    # Initial detection based on filename
    doc_info = {
        "type": "general",
        "language": "english",
        "academic_level": "undergraduate",
        "specialized_areas": []
    }
    
    # Filename-based detection
    filename_lower = document_name.lower()
    if any(term in filename_lower for term in ["histoire", "history", "revolution"]):
        doc_info["type"] = "history"
    elif any(term in filename_lower for term in ["physique", "physics", "chimie", "chemistry"]):
        doc_info["type"] = "physics"
        doc_info["specialized_areas"].append("physical_science")
    elif any(term in filename_lower for term in ["math", "calcul", "algebra"]):
        doc_info["type"] = "mathematics"
    elif any(term in filename_lower for term in ["finance", "econ", "business"]):
        doc_info["type"] = "economics"
        doc_info["academic_level"] = "professional" 
    elif any(term in filename_lower for term in ["rc", "circuit", "electronic"]):
        doc_info["type"] = "electronics"
        doc_info["specialized_areas"].append("circuits")
    
    # French language detection
    if any(term in filename_lower for term in ["francais", "france", "francaise", "examen-national"]):
        doc_info["language"] = "french"
    
    # Perform content-based detection with a sample of the content
    sample = content[:3000]  # First 3000 chars for detection
    
    # History detection
    if re.search(r'\b(century|historical|revolution|empire|dynasty|reign|king|queen|emperor|war|treaty)\b', sample, re.IGNORECASE):
        doc_info["type"] = "history"
    
    # Physics detection
    if re.search(r'\b(force|energy|mass|velocity|acceleration|voltage|current|resistance|electric|magnetic|quantum|atom|molecule|thermodynamics|mechanics)\b', sample, re.IGNORECASE):
        doc_info["type"] = "physics"
        doc_info["specialized_areas"].append("physical_science")
    
    # Electronics/circuits detection
    if re.search(r'\b(circuit|resistor|capacitor|inductor|impedance|voltage|current|ampere|ohm|farad|diode|transistor|RC|RL|RLC)\b', sample, re.IGNORECASE):
        doc_info["type"] = "electronics"
        doc_info["specialized_areas"].append("circuits")
    
    # Finance detection
    if re.search(r'\b(finance|investment|portfolio|stock|bond|asset|liability|capital|interest|dividend|market|equity|trading|profit|loss)\b', sample, re.IGNORECASE):
        doc_info["type"] = "economics"
        doc_info["specialized_areas"].append("finance")
    
    # Language detection enhancement
    if re.search(r'\b(le|la|les|du|des|un|une|est|sont|avec|pour|dans|cette|ces|qui|que|quoi|où|pourquoi|comment)\b', sample):
        doc_info["language"] = "french"
    
    # Academic level detection
    if re.search(r'\b(advanced|graduate|phd|thesis|dissertation|complex|research|analysis|theoretical|methodology)\b', sample, re.IGNORECASE):
        doc_info["academic_level"] = "graduate"
    elif re.search(r'\b(elementary|basic|simple|introduction|beginner)\b', sample, re.IGNORECASE):
        doc_info["academic_level"] = "secondary"
    
    return doc_info

def generate_exam(
    document_name: str, 
    difficulty: str = "medium", 
    exam_type: str = "mixed", 
    num_questions: int = 5
) -> Dict[str, Any]:
    """
    Generate an exam based on document content
    
    Parameters:
    - document_name: Name of the document to generate questions from
    - difficulty: 'easy', 'medium', or 'hard'
    - exam_type: 'mcq', 'essay', 'fill_blank', 'true_false', or 'mixed'
    - num_questions: Number of questions to generate
    
    Returns:
    - Dictionary containing the exam data
    """
    
    document_content = get_document_content(document_name)
    
    if not document_content:
        return {
            "success": False,
            "error": f"No content found for document: {document_name}"
        }
    
    # Detect document type to customize prompt
    doc_info = detect_document_type(document_name, document_content)
        
    difficulty_descriptions = {
        "easy": "basic recall and fundamental understanding",
        "medium": "application of concepts and moderate analysis",
        "hard": "advanced analysis, evaluation, and synthesis of complex ideas"
    }
    
    exam_type_instructions = {
        "mcq": "Create multiple-choice questions with 4 options each and only one correct answer",
        "essay": "Create in-depth essay questions that require comprehensive responses",
        "fill_blank": "Create fill-in-the-blank questions with key terms or concepts missing",
        "true_false": "Create clear true/false statements based on the content",
        "mixed": "Create a balanced mixture of different question types (MCQ, essay, fill-in-the-blank, true/false)"
    }
    
    # Customize instructions based on document type
    subject_specific_instructions = ""
    if doc_info["type"] == "history":
        subject_specific_instructions = """
        For history content:
        - Include questions about key dates, figures, events, and their significance
        - Ask about cause-and-effect relationships between historical events
        - Include questions about historical interpretations and perspectives
        - For essay questions, ask students to compare/contrast historical periods or events
        """
    elif doc_info["type"] == "physics" or doc_info["type"] == "electronics":
        subject_specific_instructions = """
        For physics/electronics content:
        - Include questions that test understanding of core concepts and principles
        - Add calculation problems where applicable (with full solutions)
        - Include questions about experimental setups and methodologies
        - Ask students to explain physical phenomena or circuit behavior
        - For complex topics, break down problems into step-by-step solutions
        """
    elif doc_info["type"] == "economics":
        subject_specific_instructions = """
        For economics/finance content:
        - Include questions about key economic theories and financial concepts
        - Add calculation problems involving financial metrics or economic indicators
        - Include case studies or scenario-based questions
        - Ask students to analyze economic trends or financial statements
        - Include questions about market mechanisms or investment strategies
        """
    
    # Language-specific instructions
    language_instructions = ""
    if doc_info["language"] == "french":
        language_instructions = """
        - Generate all questions and answers in French
        - Ensure proper French grammar, spelling, and terminology
        - Use appropriate French academic style and formatting
        """
    
    prompt = f"""
    Generate a high-quality {difficulty} level academic exam based on the following document content.
    
    Document: {document_name}
    Document Type: {doc_info["type"]}
    Academic Level: {doc_info["academic_level"]}
    Language: {doc_info["language"]}
    
    The exam should:
    1. Test {difficulty_descriptions.get(difficulty, "understanding")} of the material
    2. Contain exactly {num_questions} questions of type: {exam_type}
    3. {exam_type_instructions.get(exam_type, "Create questions that test understanding")}
    4. Be directly based on the document content, using appropriate terminology
    {subject_specific_instructions}
    {language_instructions}
    
    The exam should follow this JSON structure:
    {{
        "title": "Specific, descriptive exam title relating to the content",
        "description": "2-3 sentences describing the exam scope and objectives",
        "difficulty": "{difficulty}",
        "questions": [
            {{
                "id": 1,
                "type": "mcq|essay|fill_blank|true_false",
                "question": "Clear, well-formulated question text",
                "options": ["Option A", "Option B", "Option C", "Option D"], 
                "answer": "Correct answer or answer outline",
                "explanation": "Detailed explanation of why this is the correct answer",
                "points": 5
            }},
            ...
        ]
    }}
    
    IMPORTANT INSTRUCTIONS:
    1. For MCQ questions:
       - Provide 4 options with only one correct answer
       - Make all distractors plausible but clearly incorrect
       - Label the correct answer clearly
    
    2. For Essay questions:
       - Include a clear prompt that requires analysis
       - Provide an expected answer outline with key points that should be covered
       - Include evaluation criteria
    
    3. For Fill-in-the-blank questions:
       - Use underscores for blanks
       - Ensure the missing words are significant terms from the content
       - Provide the exact missing words in the answer
    
    4. For True/False questions:
       - Create unambiguous statements
       - Provide a clear explanation for why the statement is true or false
    
    5. If type is "mixed":
       - Ensure a balanced mix of all question types
       - Include at least one of each question type
    
    6. For all questions:
       - Use clear academic language appropriate for the subject
       - Directly relate to specific content in the document
       - Avoid opinion-based questions unless explicitly marked as such
       - Provide detailed answer explanations that can serve as learning material
    
    Return ONLY the valid JSON object representing the exam with no additional text.
    
    Here's the document content to base the exam on:
    {document_content[:15000]}  
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1500,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        response_text = response.text
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            exam_data = json.loads(json_str)
        except:
            exam_data = extract_json_from_text(response_text)
        
        # Verify required fields and question count
        if not exam_data.get("questions") or len(exam_data.get("questions", [])) < num_questions * 0.7:
            # Try again with a more specific prompt if we received too few questions
            return retry_exam_generation(document_name, document_content, difficulty, exam_type, num_questions, doc_info)
        
        # Add source document and success indicator
        exam_data["source_document"] = document_name
        exam_data["success"] = True
        
        return exam_data
    
    except Exception as e:
        print(f"Error generating exam: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate exam: {str(e)}",
            "partial_response": response.text if 'response' in locals() else "No response generated"
        }

def retry_exam_generation(document_name, document_content, difficulty, exam_type, num_questions, doc_info):
    """Retry generating an exam with a more focused prompt"""
    
    # Language-specific instructions
    language_instructions = ""
    if doc_info["language"] == "french":
        language_instructions = """
        - Generate all questions and answers in French
        - Ensure proper French grammar, spelling, and terminology
        """
    
    retry_prompt = f"""
    I need a professional {difficulty} level academic exam with exactly {num_questions} questions of type {exam_type}.
    The exam must be based exclusively on this document content and follow academic standards.
    
    Document: {document_name}
    Document Type: {doc_info["type"]}
    Academic Level: {doc_info["academic_level"]}
    Language: {doc_info["language"]}
    
    {language_instructions}
    
    Format as a valid JSON object with this structure exactly:
    {{
        "title": "Descriptive exam title",
        "description": "Clear exam description and objectives",
        "difficulty": "{difficulty}",
        "questions": [
            {{
                "id": 1,
                "type": "{exam_type}",
                "question": "Well-formulated question text",
                "options": ["Option A", "Option B", "Option C", "Option D"], 
                "answer": "Correct answer with explanation",
                "explanation": "Detailed explanation of the correct answer",
                "points": 5
            }},
            // MORE QUESTIONS TO REACH {num_questions} TOTAL
        ]
    }}
    
    If type is "mixed", include a balance of mcq, essay, fill_blank, and true_false questions.
    Ensure all questions are academically rigorous and directly related to this content.
    
    Document content:
    {document_content[:10000]}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(retry_prompt, temperature=0.5)
        response_text = response.text
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            exam_data = json.loads(json_str)
        except:
            exam_data = extract_json_from_text(response_text)
        
        exam_data["source_document"] = document_name
        exam_data["success"] = True
        
        return exam_data
        
    except Exception as e:
        print(f"Error in retry exam generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate exam after retry: {str(e)}"
        }

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from text that may contain non-JSON elements"""
    try:
        
        import re
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            return {
                "title": "Generated Exam",
                "description": "Exam generated from document content",
                "questions": [],
                "error": "Failed to parse exam structure"
            }
    except:
        return {
            "title": "Generated Exam",
            "description": "Exam generated from document content",
            "questions": [],
            "error": "Failed to parse exam structure"
        }

def generate_exercise(
    document_name: str,
    exercise_type: str = "practice",
    difficulty: str = "medium",
    focus_area: str = None
) -> Dict[str, Any]:
    """
    Generate a practice exercise based on document content
    
    Parameters:
    - document_name: Name of the document to generate exercises from
    - exercise_type: 'practice', 'worksheet', or 'challenge'
    - difficulty: 'easy', 'medium', or 'hard'
    - focus_area: Optional specific topic/concept to focus on
    
    Returns:
    - Dictionary containing the exercise data
    """
    document_content = get_document_content(document_name)
    
    if not document_content:
        return {
            "success": False,
            "error": f"No content found for document: {document_name}"
        }
    
    # Detect document type to customize prompt
    doc_info = detect_document_type(document_name, document_content)
    
    exercise_type_instructions = {
        "practice": "Create applied practice problems that directly test understanding of concepts",
        "worksheet": "Design a structured worksheet with a mix of question types and learning activities",
        "challenge": "Develop complex, critical thinking exercises that require deeper analysis and synthesis"
    }
    
    difficulty_descriptions = {
        "easy": "basic recall and fundamental understanding",
        "medium": "application of concepts and moderate analysis",
        "hard": "advanced analysis, evaluation, and synthesis of complex ideas"
    }
    
    # Customize instructions based on document type
    subject_specific_instructions = ""
    if doc_info["type"] == "history":
        subject_specific_instructions = """
        For history exercises:
        - Include timeline activities and chronological ordering exercises
        - Add primary source analysis components
        - Include map-based activities where relevant
        - Create debate or discussion components on historical interpretations
        - Add comparative analysis between historical periods or regions
        """
    elif doc_info["type"] == "physics" or doc_info["type"] == "electronics":
        subject_specific_instructions = """
        For physics/electronics exercises:
        - Include both theoretical problems and practical calculations
        - Add diagram-based analysis components
        - Include experimental design or analysis components
        - Create step-by-step problems with increasing complexity
        - Add real-world application scenarios
        - Include circuit analysis or design problems where relevant
        """
    elif doc_info["type"] == "economics":
        subject_specific_instructions = """
        For economics/finance exercises:
        - Include data analysis and interpretation components
        - Add case study analysis exercises
        - Include calculation problems involving financial metrics
        - Create market scenario analysis components
        - Add policy analysis or recommendation exercises
        """
    
    # Language-specific instructions
    language_instructions = ""
    if doc_info["language"] == "french":
        language_instructions = """
        - Generate all exercise content and solutions in French
        - Ensure proper French terminology, grammar, and spelling
        - Use appropriate French academic style and formatting
        """
    
    focus_query = ""
    if focus_area:
        focus_query = f"Focus specifically on the topic of '{focus_area}'."
    
    prompt = f"""
    Create a comprehensive {difficulty} difficulty {exercise_type} exercise based on the following document content. {focus_query}
    
    Document: {document_name}
    Document Type: {doc_info["type"]}
    Academic Level: {doc_info["academic_level"]}
    Language: {doc_info["language"]}
    
    {exercise_type_instructions.get(exercise_type, "")}
    {subject_specific_instructions}
    {language_instructions}
    
    The exercise should follow this JSON structure:
    {{
        "title": "Exercise title based on document content",
        "description": "Detailed description of the exercise's purpose and learning objectives",
        "difficulty": "{difficulty}",
        "estimated_time": "Time estimation (e.g., '30 minutes')",
        "components": [
            {{
                "type": "explanation|problem|activity|reflection|calculation|analysis",
                "title": "Component title",
                "content": "Detailed content of this component with clear instructions",
                "instructions": "Step-by-step instructions for completing this component"
            }},
            ...
        ],
        "solution_guide": "Comprehensive guidance or solutions for the exercise components"
    }}
    
    IMPORTANT INSTRUCTIONS:
    1. For a {exercise_type} exercise at {difficulty} level ({difficulty_descriptions.get(difficulty, "")}):
       - Include at least 5-7 substantive components
       - Each component should have clear instructions and detailed content
       - Vary the component types to test different skills (understanding, application, analysis)
    2. Make the exercises realistic, practical and directly relevant to the document content
    3. Ensure the solution guide provides detailed explanations and complete answers
    4. For mathematical or scientific content, include properly formatted problems and solutions
    5. For history or textual content, include appropriate analytical frameworks and evidence-based reasoning
    6. Return ONLY the valid JSON object with no additional text
    
    Here's the document content to base the exercise on:
    {document_content[:15000]}  
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2000,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        response_text = response.text
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            exercise_data = json.loads(json_str)
        except:
            exercise_data = extract_json_from_text(response_text)
            
        # Verify required fields
        if not exercise_data.get("components") or len(exercise_data.get("components", [])) < 3:
            # Try again with a more specific prompt
            return retry_exercise_generation(document_name, document_content, exercise_type, difficulty, focus_area, doc_info)
        
        exercise_data["source_document"] = document_name
        exercise_data["success"] = True
        
        return exercise_data
    
    except Exception as e:
        print(f"Error generating exercise: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate exercise: {str(e)}",
            "partial_response": response.text if 'response' in locals() else "No response generated"
        }

def retry_exercise_generation(document_name, document_content, exercise_type, difficulty, focus_area, doc_info):
    """Retry generating an exercise with a more focused prompt"""
    
    # Language-specific instructions
    language_instructions = ""
    if doc_info["language"] == "french":
        language_instructions = """
        - Generate all exercise content and solutions in French
        - Ensure proper French terminology, grammar, and spelling
        """
    
    retry_prompt = f"""
    I need a high-quality {difficulty} {exercise_type} exercise based on this document content.
    
    Document: {document_name}
    Document Type: {doc_info["type"]}
    Academic Level: {doc_info["academic_level"]}
    Language: {doc_info["language"]}
    
    {language_instructions}
    
    The exercise MUST have at least 5 well-structured components with clear titles, detailed content,
    and specific instructions for each component. The exercise should follow academic standards.
    
    Format as a valid JSON object with this structure:
    {{
        "title": "Descriptive title",
        "description": "2-3 sentence overview of the exercise and its learning objectives",
        "difficulty": "{difficulty}",
        "estimated_time": "30-45 minutes",
        "components": [
            {{
                "type": "explanation",
                "title": "Understanding [Key Concept]",
                "content": "Detailed explanation of concept with example",
                "instructions": "Read the following explanation and answer questions below"
            }},
            {{
                "type": "problem",
                "title": "Applying [Concept]",
                "content": "Problem statement with all necessary details",
                "instructions": "Follow these steps to solve the problem"
            }},
            // MORE COMPONENTS (AT LEAST 5 IN TOTAL)
        ],
        "solution_guide": "Step-by-step solutions for all components with detailed explanations"
    }}
    
    {f"Focus specifically on: {focus_area}" if focus_area else ""}
    
    Document content:
    {document_content[:10000]}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(retry_prompt, temperature=0.5)
        response_text = response.text
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            exercise_data = json.loads(json_str)
        except:
            exercise_data = extract_json_from_text(response_text)
        
        exercise_data["source_document"] = document_name
        exercise_data["success"] = True
        
        return exercise_data
        
    except Exception as e:
        print(f"Error in retry exercise generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate exercise after retry: {str(e)}"
        }