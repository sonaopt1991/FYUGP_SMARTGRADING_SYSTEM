from flask import Flask, request, render_template, redirect, url_for
import os
import re
import pandas as pd
import requests

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluated_data = []  # Store feedback dynamically instead of using a CSV


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
    global evaluated_data
    evaluated_data = []

    df = pd.read_csv(file_path).dropna(subset=["Question", "Response"], how="all")

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        student_code = str(row.get("Response", "")).strip()
        if not question or not student_code:
            continue  # Skip rows with missing values

        feedback, score = evaluate_student_code(question, student_code)
        evaluated_data.append({"question": question, "student_code": student_code, "feedback": feedback, "score": score})


@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle file uploads and trigger processing."""
    global evaluated_data
    evaluated_data = []  # Reset evaluated data on new upload

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
    """Display evaluation results."""
    if not evaluated_data:
        return redirect(url_for("upload_file"))
    return render_template("feedback22.html", feedback_data=evaluated_data)


if __name__ == "__main__":
    app.run(debug=True)
