from fpdf import FPDF
import os
import json
from datetime import datetime
import tempfile

class EnhancedPDF(FPDF):
    """Enhanced PDF class with better styling but reliable generation"""
    
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.set_auto_page_break(True, margin=15)
        self.set_margins(20, 20, 20)
        self._current_question_number = 0
        self.title = ""
        self.header_bg_color = (70, 130, 180)  
        self.title_color = (25, 25, 112)       
        self.question_color = (0, 51, 102)   
        
    def header(self):
        """Header with light background"""
        self.set_font('Arial', 'B', 12)
        
        self.set_fill_color(*self.header_bg_color)
        self.set_text_color(255, 255, 255)  
        self.cell(0, 12, self.title, 0, 1, 'C', True)
        
        self.set_text_color(0, 0, 0)
        self.ln(5)
        
    def footer(self):
        """Footer with page number"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*self.title_color)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.set_text_color(0, 0, 0)  
        
    def add_section_title(self, title):
        """Add a nicer section title with line underneath"""
        self.ln(8)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.title_color)
        self.cell(0, 10, title, 0, 1, 'L')
        
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        
        self.set_text_color(0, 0, 0)
        self.ln(5)
        
    def add_text(self, text):
        """Add text with proper wrapping"""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, str(text))
        self.ln(3)

    def draw_circle(self, x, y, r, style='D'):
        """Draw a circle with center(x,y) and radius r"""
        self.ellipse(x, y, r, r, style)
        
    def add_question(self, question_text, question_type="", options=None):
        """Add a question with better formatting and answer spaces"""
        self._current_question_number += 1
        
        if self.get_y() > self.h - 40 and self.get_y() > 40:
            self.add_page()
        
        self.set_fill_color(*self.question_color)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 11)
        
        circle_y = self.get_y() + 4
        self.draw_circle(self.l_margin + 5, circle_y, 4, 'F')
        self.set_xy(self.l_margin + 1, circle_y - 2)
        self.cell(8, 4, str(self._current_question_number), 0, 0, 'C')
        
        question_x = self.l_margin + 12
        self.set_text_color(*self.question_color)
        self.set_xy(question_x, circle_y - 4)
        
        if question_type:
            self.set_font('Arial', 'I', 9)
            self.cell(30, 8, f"({question_type})", 0, 0)
            question_x += 30
        
        self.set_text_color(0, 0, 0)
        
        self.set_xy(question_x, circle_y - 4)
        self.set_font('Arial', '', 11)
        self.multi_cell(self.w - question_x - self.r_margin, 6, str(question_text))
        self.ln(3)
        
        
        if options and len(options) > 0:
            self.ln(2)
            for i, option in enumerate(options):
                self.set_x(self.l_margin + 15)
                option_letter = chr(65 + i)  
                
                
                self.draw_circle(self.l_margin + 10, self.get_y() + 3, 3, 'D')
                
                
                self.set_xy(self.l_margin + 15, self.get_y())
                self.set_font('Arial', '', 11)
                self.cell(10, 6, f"{option_letter})", 0, 0)
                self.multi_cell(self.w - self.l_margin - 25, 6, str(option))
                self.ln(2)
        
        
        elif question_type in ["essay", "fill_blank"]:
            self.ln(3)
            answer_space_height = 30 if question_type == "essay" else 15
            
           
            self.set_fill_color(245, 245, 245)
            answer_box_y = self.get_y()
            self.rect(self.l_margin + 15, answer_box_y, 
                     self.w - self.l_margin - self.r_margin - 15, answer_space_height, 'F')
            
          
            if question_type == "essay":
                for line in range(4):
                    line_y = answer_box_y + 7 + (line * 6)
                    self.line(self.l_margin + 20, line_y, 
                             self.w - self.r_margin - 20, line_y)
                             
            self.set_y(answer_box_y + answer_space_height + 2)
        
        
        elif question_type == "true_false":
            self.ln(3)
            self.set_x(self.l_margin + 20)
            
            
            self.draw_circle(self.l_margin + 25, self.get_y() + 3, 3, 'D')
            self.set_xy(self.l_margin + 30, self.get_y())
            self.cell(20, 6, "True", 0, 0)
            
        
            self.draw_circle(self.l_margin + 65, self.get_y() + 3, 3, 'D') 
            self.set_xy(self.l_margin + 70, self.get_y())
            self.cell(20, 6, "False", 0, 1)
        
       
        self.ln(5)
        
        
        self.set_draw_color(200, 200, 200)  
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_draw_color(0, 0, 0)
        self.ln(5)

    def add_instructions_box(self, instructions):
        """Add instructions in a nice looking box"""
        self.set_fill_color(240, 240, 240)  
        self.rect(self.l_margin, self.get_y(), self.w - 2*self.r_margin, 30, 'F')
        
        
        self.set_xy(self.l_margin + 5, self.get_y() + 5)
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, "Instructions:", 0, 1)
        
        
        self.set_xy(self.l_margin + 5, self.get_y())
        self.set_font('Arial', '', 10)
        self.multi_cell(self.w - 2*self.r_margin - 10, 5, str(instructions))
        
        self.set_y(self.get_y() + 5)  

def safe_str(value):
    """Convert any value to string safely"""
    if value is None:
        return ""
    return str(value)

def generate_exam_pdf(exam_data, include_answers=False):
    """Generate a nicer PDF from exam data"""
    try:
        pdf = EnhancedPDF()
        pdf.add_page()
        
        
        title = safe_str(exam_data.get('title', 'Generated Exam'))
        pdf.title = title
        
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        
        if 'description' in exam_data:
            pdf.set_font('Arial', 'I', 11)
            pdf.multi_cell(0, 6, safe_str(exam_data['description']))
            pdf.ln(5)
            
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Difficulty: {safe_str(exam_data.get('difficulty', 'medium'))}", 0, 1)
        pdf.cell(0, 5, f"Source: {safe_str(exam_data.get('source_document', ''))}", 0, 1)
        pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.ln(5)
        
        
        instructions = (
            "Read each question carefully. Answer all questions to the best of your ability. "
            "For multiple choice questions, select the best answer by marking the circle. "
            "For fill-in-the-blank questions, write your answer in the space provided. "
            "For essay questions, provide a complete response in the provided area."
        )
        pdf.add_instructions_box(instructions)
        pdf.ln(10)
        
        
        pdf.add_section_title("Questions")
        
        questions = exam_data.get('questions', [])
        for question in questions:
            q_type = safe_str(question.get('type', ''))
            q_text = safe_str(question.get('question', ''))
            
            
            options = None
            if 'options' in question and question['options']:
                options = [safe_str(opt) for opt in question['options']]
            
            pdf.add_question(q_text, question_type=q_type, options=options)
        
        
        if include_answers:
            pdf.add_page()
            pdf.add_section_title("Answer Key")
            
            for i, question in enumerate(questions):
                answer = safe_str(question.get('answer', 'No answer provided'))
                explanation = safe_str(question.get('explanation', ''))
                
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, f"Question {i+1}", 0, 1)
                
                pdf.set_font('Arial', '', 11)
                pdf.cell(0, 6, f"Answer: {answer}", 0, 1)
                
                if explanation:
                    pdf.set_font('Arial', 'I', 10)
                    pdf.multi_cell(0, 6, f"Explanation: {explanation}")
                    
                pdf.ln(5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
            
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

def generate_exercise_pdf(exercise_data, include_solutions=False):
    """Generate an enhanced PDF from exercise data"""
    try:
        pdf = EnhancedPDF()
        pdf.add_page()
        
       
        title = safe_str(exercise_data.get('title', 'Generated Exercise'))
        pdf.title = title
        
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        
        if 'description' in exercise_data:
            pdf.set_font('Arial', 'I', 11)
            pdf.multi_cell(0, 6, safe_str(exercise_data['description']))
            
        
        pdf.ln(5)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Difficulty: {safe_str(exercise_data.get('difficulty', 'medium'))}", 0, 1)
        pdf.cell(0, 5, f"Estimated time: {safe_str(exercise_data.get('estimated_time', '30 minutes'))}", 0, 1)
        pdf.cell(0, 5, f"Source: {safe_str(exercise_data.get('source_document', ''))}", 0, 1)
        pdf.ln(10)
        
        
        components = exercise_data.get('components', [])
        
        for i, component in enumerate(components):
            
            if pdf.get_y() > pdf.h - 60 and pdf.get_y() > 40:
                pdf.add_page()
                
            component_title = safe_str(component.get('title', f'Component {i+1}'))
            component_type = safe_str(component.get('type', ''))
            
            pdf.add_section_title(f"{i+1}. {component_title} ({component_type})")
            
            content = safe_str(component.get('content', ''))
            pdf.add_text(content)
            
            if 'instructions' in component and component['instructions']:
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, "Instructions:", 0, 1)
                pdf.add_text(safe_str(component['instructions']))
                
                
                pdf.ln(5)
                pdf.set_fill_color(245, 245, 245)  
                pdf.rect(pdf.l_margin + 10, pdf.get_y(), 
                         pdf.w - pdf.l_margin - pdf.r_margin - 20, 30, 'F')
                pdf.set_y(pdf.get_y() + 35)
                
            pdf.ln(8)
        
        if include_solutions and 'solution_guide' in exercise_data:
            pdf.add_page()
            pdf.add_section_title("Solution Guide")
            pdf.add_text(safe_str(exercise_data['solution_guide']))
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
            
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None