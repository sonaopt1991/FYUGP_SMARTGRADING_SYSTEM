import os
import re
import pandas as pd
import requests
from flask import Flask, request, render_template, send_file, redirect, url_for

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

evaluated_file_path = os.path.join(OUTPUT_FOLDER, "evaluated_responses.csv")


def extract_score(feedback):
    """Extract numeric score (out of 10) only if 'Score' or 'Overall Score' is mentioned in the feedback."""
    match = re.search(r'\b(?:Overall Score|Score):\s*(\d{1,2})/10', feedback, re.IGNORECASE)
    if match:
        return min(max(int(match.group(1)), 0), 10)
    return 0  # Return None if no valid score format is found
# def extract_score(feedback):
#     """Extract numeric score (out of 10) from feedback, including 'Overall Score: X/10'."""
    
#     # First, try to extract **Overall Score: 7/10** or similar
#     #match = re.search(r'Overall Score:\s*(\d{1,2})/10', feedback, re.IGNORECASE)
#     #match = re.search(r'Overall Score:\s*(\d{1,2})/10', feedback, re.IGNORECASE)
#     match = re.search(r'\bScore:\s*(\d{1,2})/10', feedback, re.IGNORECASE)
#     if match:
#         score = int(match.group(1))
#         return min(max(score, 0), 10)

#     # If no "Overall Score" found, fall back to any plain number (old logic)
#     match = re.search(r'(\b\d{1,2}\b)', feedback)
#     if match:
#         score = int(match.group(1))
#         return min(max(score, 0), 10)

#     # Default score if no valid number found
#     return 5



# Function to evaluate student code using Ollama (Gemma)
def evaluate_student_code(question, student_code):
    """Call Ollama's Gemma model for feedback & score."""
    prompt = f"""
    You are a coding evaluator. Check the student's code for:
    - Correctness (Does it solve the problem?)
    - Efficiency (Big-O notation)
    - Readability (Best practices & style)
    - Possible improvements

    Question: {question}
    Student's Code:
    {student_code}

    Provide structured feedback. Finally, give an **overall score out of 10** (no explanation needed).
    """

    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to get response from Ollama: {response.text}")

    feedback = response.json().get('response', '').strip()
    score = extract_score(feedback)

    return feedback, score


def process_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Remove completely empty rows
    df = df.dropna(subset=["Question", "Response"], how="all")
    
    results = []
    total_score = 0

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        student_code = str(row.get("Response", "")).strip()
        
        # Skip empty rows properly
        if not question or not student_code:
            continue  

        feedback, score = evaluate_student_code(question, student_code)
        total_score += score
        results.append({
            "Question": question, 
            "Student Code": student_code, 
            "Feedback": feedback, 
            "Score (Out of 10)": score
        })

    # Ensure at least one row is present to prevent errors
    if not results:
        return pd.DataFrame(columns=["Question", "Student Code", "Feedback", "Score (Out of 10)"])

    #  Append total score row
    total_row = {"Question": "Total Score", "Student Code": "", "Feedback": "", "Score (Out of 10)": total_score}
    output_df = pd.DataFrame(results + [total_row])  # Combine results and total row
    
    output_df.to_csv(evaluated_file_path, index=False)  # Save filtered results
    return output_df
    

# Route: Upload page + Process file
@app.route("/", methods=["GET", "POST"])
def upload_file():
    global feedback_data  # To pass data to /results
    feedback_data = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        feedback_data = process_csv(file_path).to_dict(orient="records")
        return redirect(url_for("results"))

    return render_template("index.html")


# Route: Display evaluated responses in table
@app.route("/results")
def results():
    if feedback_data is None:
        return redirect(url_for("upload_file"))
    return render_template("results.html", feedback_data=feedback_data)


# Route: Download evaluated file
@app.route("/download")
def download():
    if os.path.exists(evaluated_file_path):
        return send_file(evaluated_file_path, as_attachment=True)
    return "No evaluated file available", 404


if __name__ == "__main__":
    app.run(debug=True)
