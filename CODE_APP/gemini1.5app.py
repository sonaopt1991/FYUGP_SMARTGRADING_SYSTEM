import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure Gemini API Key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# Load CSV file (update path if necessary)
response = pd.read_csv('student_responses_all_1.csv')

def evaluate_code(question, student_code):
    """Generates AI feedback for the student's code using Google Gemini 1.5 Pro."""
    prompt = f"""
    You are an expert coding instructor. Analyze the following code based on:
    1. Correctness (Does it solve the problem?)
    2. Efficiency (Big-O notation)
    3. Code readability (Best practices & style)
    4. Alternative improvements

    Question: {question}
    Student's Code:
    {student_code}
   
    Provide structured feedback in a clear and structured format.make it short and simple.
    After your analysis, please provide an **overall score out of 10** for the student's code.no explanation needed.
    """

    
    model = genai.GenerativeModel('gemini-1.5-pro')  
    response = model.generate_content(prompt)

    return response.text

@app.route('/')
def index():
    """Route to display student responses and feedback."""
    feedback_data = []

    for _, row in response.iterrows():
        question = row['Question']
        student_code = row['Response']

        try:
            feedback = evaluate_code(question, student_code)
        except Exception as e:
            feedback = f"Error generating feedback: {str(e)}"

        feedback_data.append({
            "Question": question,
            "student_code": student_code,
            "feedback": feedback
        })

    return render_template("index.html", feedback_data=feedback_data)

if __name__ == '__main__':
    app.run(debug=True)
