from flask import Flask, request, render_template, redirect, url_for
import os
import re
import pandas as pd
import requests

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluated_data = []  # Store feedback dynamically instead of CSV


def extract_score(feedback):
    """Extract numeric score (out of 10) from feedback."""
    match = re.search(r'\b(?:Overall Score|Score):\s*(\d{1,2})/10', feedback, re.IGNORECASE)
    if match:
        return min(max(int(match.group(1)), 0), 10)
    return 0  # Default to 0 if no score found


def evaluate_student_code(question, student_code):
    """Call Ollama's Gemma model for feedback & score."""
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

    Provide structured feedback. Finally, give an **overall score out of 10** (no explanation needed).
    """

    ollama_url = "http://localhost:11434/api/generate"
    payload = {"model": "gemma:2b", "prompt": prompt, "stream": False}

    response = requests.post(ollama_url, json=payload)
    if response.status_code != 200:
        return "Failed to get response from Ollama.", 0

    feedback = response.json().get('response', '').strip()
    score = extract_score(feedback) or 5

    return feedback, score


def process_csv(file_path):
    global evaluated_data
    
    df = pd.read_csv(file_path).dropna(subset=["Question", "Response"], how="all")

    evaluated_data = []
    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        student_code = str(row.get("Response", "")).strip()
        if not question or not student_code:
            continue  

        feedback, score = evaluate_student_code(question, student_code)
        evaluated_data.append({"question": question, "student_code": student_code, "feedback": feedback, "score": score})


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global evaluated_data
    evaluated_data = []

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        process_csv(file_path)
        return redirect(url_for("results"))

    return render_template("index.html")


@app.route("/results")
def results():
    if not evaluated_data:
        return redirect(url_for("upload_file"))
    return render_template("feedback.html", feedback_data=evaluated_data)


if __name__ == "__main__":
    app.run(debug=True)
