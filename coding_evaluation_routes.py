from flask import Blueprint, request, render_template, redirect, url_for, session
import os
import re
import pandas as pd
import requests

# Define the blueprint
coding_evaluation_bp = Blueprint('coding_evaluation', __name__, template_folder='templates')

# Directory to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluated_data = []  # Store feedback dynamically


def extract_score(feedback):
    """Extract numeric score (out of 10) from feedback text."""
    match = re.search(r'\b(?:Overall Score|Score):\s*(\d{1,2})/10', feedback, re.IGNORECASE)
    return min(max(int(match.group(1)), 0), 10) if match else 5  # Default to 5 if no score found


def evaluate_student_code(question, student_code):
    """Call Ollama's Gemma model for structured feedback & score."""
    prompt = f"""
    You are a coding evaluator. Check the student's code for:
    - **Correctness** (Does it solve the problem?)
    - **Efficiency** (Big-O notation)
    - **Readability** (Best practices & style)
    - **Possible improvements**

    **Question:** {question}
    **Student's Code:**
    ```python
    {student_code}
    ```

    Provide structured feedback. Finally, **MANDATORILY** include this exact format:
    **Overall Score: X/10** (Replace X with a number between 0-10)
    """

    ollama_url = "http://localhost:11434/api/generate"
    payload = {"model": "gemma:2b", "prompt": prompt, "stream": False}

    try:
        response = requests.post(ollama_url, json=payload)
        if response.status_code == 200:
            feedback = response.json().get('response', '').strip()
        else:
            feedback = "Error: Unable to fetch response from the model."
    except Exception as e:
        feedback = f"Error connecting to model: {str(e)}"

    score = extract_score(feedback)
    return feedback, score


def process_csv(file_path):
    """Read the uploaded CSV and evaluate each student's response."""
    from combined_app import store_scores  # Import inside function to avoid circular import
    global evaluated_data
    #evaluated_data = []  # Reset evaluated data

    df = pd.read_csv(file_path).dropna(subset=["Question", "Response"], how="all")

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        student_code = str(row.get("Response", "")).strip()
        if not question or not student_code:
            continue  # Skip rows with missing values

        feedback, score = evaluate_student_code(question, student_code)
        evaluated_data.append({"question": question, "student_code": student_code, "feedback": feedback, "score": score})


@coding_evaluation_bp.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle file uploads and trigger processing."""
    from combined_app import store_scores  # Import inside function to avoid circular import
    global evaluated_data
    evaluated_data = []  # Reset evaluated data on new upload
    total_score=0
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        process_csv(file_path)

        if 'username' in session:  
            username = session['username']  # Get the username from session
            score_file = "score_df.csv"
            # Path to your score DataFrame
            for eval_data in evaluated_data:
                print(f"Storing score for question: {eval_data['question']}, score: {eval_data['score']}")
                code_score = eval_data["score"]
                total_score += code_score
                # Use store_scores to store MCQ, Essay, and Code scores along with the Total Score
                store_scores(username, mcq_score=None, essay_score=None, Code_score=total_score, score_file=score_file)

        return redirect(url_for("coding_evaluation.results"))

    return render_template("index.html")


@coding_evaluation_bp.route("/results")
def results():
    """Display evaluation results."""
    if not evaluated_data:
        return redirect(url_for("coding_evaluation.upload_file"))
    return render_template("feedback22.html", feedback_data=evaluated_data)
