from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import nltk
import os
from transformers import AutoTokenizer, AutoModel
import torch
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch 
from model import evaluate_student_performance,generate_student_performance_bar_chart#train_and_generate_results, generate_streamwise_plot,evaluate_student_performance,generate_bar_plot
from flask import Blueprint, render_template, session, redirect, url_for, flash,request, send_file,session
import pickle
#from faculty_routes import faculty_bp
from faculty_allplots import faculty1_bp
from coding_evaluation_routes import coding_evaluation_bp
from plots import generate_top_students_bar_chart, generate_student_distribution_box_plot
# Load the model 
model = pickle.load(open('model.pkl', 'rb'))
# Initialize the Flask App
app = Flask(__name__)
app.secret_key = '123545'
# Register the blueprint
app.register_blueprint(coding_evaluation_bp, url_prefix='/coding_evaluation')

# Register Faculty Blueprint

app.register_blueprint(faculty1_bp, url_prefix='/faculty1')
score_file = "score_df.csv"

# Ensure score_df is always available
if os.path.exists(score_file):
    score_df = pd.read_csv(score_file, index_col="Reg.No.")
else:
    score_df = pd.DataFrame(columns=["Reg.No.", "MCQ Score", "Essay Score", "Code Score","Total Score"])
    score_df.to_csv(score_file, index=False)
    
# Load models for essay test
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Load data for MCQ test
credentials_df = pd.read_csv("USERS.csv")
questions_df_mcq = pd.read_excel("Book2.xlsx")  # MCQ questions and options
#questions_df_mcq = pd.read_csv("python_mcq_questions.csv")  # MCQ questions and options

questions_df_essay = pd.read_csv('essayquestion.csv')  # Essay test questions and model answers

# Preprocessing functions for essay test
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return set(tokens)

def compute_tfidf_similarity(model_answers, student_answer):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(model_answers + [student_answer])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    return similarities

def compute_transformer_similarity(model_answers, student_answer):
    def get_embedding(text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / summed_mask

    student_embedding = get_embedding(student_answer)
    similarities = [torch.cosine_similarity(student_embedding, get_embedding(ans)).item() for ans in model_answers]
    return similarities

# MCQ Grader class for objective test
class Grader:
    def __init__(self, questions_data):
        self.questions_data = questions_data
        self.correct_answers = self.load_correct_answers()
        self.total_questions = len(self.questions_data)

    def load_questions(self):
        return self.questions_data['Question'].tolist()

    def load_options(self):
        options = {}
        for _, row in self.questions_data.iterrows():
            options[row['Question']] = {
                'A': row['Option A'],
                'B': row['Option B'],
                'C': row['Option C'],
                'D': row['Option D']
            }
        return options

    def load_correct_answers(self):
        return dict(zip(self.questions_data['Question'], self.questions_data['Correct Option']))

    def grade_answers(self, user_answers):
        score = sum(1 for q, ans in user_answers.items() if self.correct_answers.get(q) == ans)
        return score, self.total_questions

# Store user scores for both MCQ and Essay
def store_scores(username, mcq_score=None, essay_score=None,Code_score=None, score_file="score_df.csv"):
    try:
        scores_df = pd.read_csv(score_file)
    except FileNotFoundError:
        scores_df = pd.DataFrame(columns=["Reg.No.", "MCQ Score", "Essay Score","Code Score","Total Score"])

    if username in scores_df["Reg.No."].values:
        if mcq_score is not None:
            scores_df.loc[scores_df["Reg.No."] == username, "MCQ Score"] = mcq_score
        if essay_score is not None:
            scores_df.loc[scores_df["Reg.No."] == username, "Essay Score"] = essay_score
        if Code_score is not None:
            scores_df.loc[scores_df["Reg.No."] == username, "Code Score"] = Code_score
    else:
        new_row = pd.DataFrame({"Reg.No.": [username], "MCQ Score": [mcq_score], "Essay Score": [essay_score],"Code Score": [Code_score], "Total Score": [0] })
        scores_df = pd.concat([scores_df, new_row], ignore_index=True)
    # Calculate Total Score (sum of MCQ, Essay, and Code scores)
    scores_df["Total Score"] = scores_df["MCQ Score"].fillna(0) + scores_df["Essay Score"].fillna(0) + scores_df["Code Score"].fillna(0)

    scores_df.to_csv(score_file, index=False)
@app.route('/')
def home():
    return render_template('landing_page.html')

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        credentials = dict(zip(credentials_df['Reg.No.'], credentials_df['Password']))
        
        if username in credentials and credentials[username] == password:
            session['username'] = username
            return redirect(url_for('student_dashboard'))
        else:
            flash("Invalid credentials! Try again.")
    return render_template('student_login.html')
    

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove user from session
    flash("You have been logged out.")
    # return redirect(url_for('student_login'))
    return redirect(url_for('home')) 


@app.route('/student_dashboard')
def student_dashboard():
    if 'username' not in session:
        return redirect(url_for('student_login'))
    return render_template('stddashboard.html')

@app.route('/mcq', methods=['GET', 'POST'])
def mcq():
    if 'username' not in session:
        return redirect(url_for('student_login'))
    
    grader = Grader(questions_df_mcq)
    questions = grader.load_questions()
    options = grader.load_options()
    
    mcq_score = None  # To store the score after submission
    feedback = None   # To store feedback after submission

    if request.method == 'POST':
        user_answers = {q: request.form[q] for q in questions}
        mcq_score, total_questions = grader.grade_answers(user_answers)

        # Store the score
        store_scores(session['username'], mcq_score=mcq_score)

        # Generate feedback based on score
        percentage = (mcq_score / total_questions) * 100
        if percentage >= 90:
            feedback = "Excellent performance! Keep up the great work. 🎉"
        elif percentage >= 75:
            feedback = "Good job! You have a strong understanding of the concepts. ✅"
        elif percentage >= 50:
            feedback = "Fair attempt! Revise the topics to improve further. 📚"
        else:
            feedback = "Needs improvement. Consider reviewing the study material and practicing more. 💡"

    return render_template('quiz.html', questions=questions, options=options, mcq_score=mcq_score, feedback=feedback)

@app.route('/upload', methods=['GET', 'POST'])
def upload_response():
    if 'username' not in session:
        return redirect(url_for('student_login'))
    
    questions = questions_df_essay['Question'].tolist()
    
    if request.method == 'POST':
        file = request.files['response_file']
        if file:
            file.save(os.path.join('uploads', file.filename))
            return redirect(url_for('grade_essay', filename=file.filename))
    
    return render_template('upload1.html', questions=questions)

@app.route('/view_results')
def view_results():
    if 'username' not in session:
        return redirect(url_for('student_login'))
    
    reg_no = session['username']
    scores_df = pd.read_csv("score_df.csv")
    student_scores = scores_df[scores_df["Reg.No."] == reg_no]
    
    if student_scores.empty:
        flash("No results found for this student.")
        return redirect(url_for('student_dashboard'))
    
    mcq_score = student_scores["MCQ Score"].fillna(0).astype(int).values[0]
    essay_score = student_scores["Essay Score"].fillna(0).astype(int).values[0]
    code_score=student_scores["Code Score"].fillna(0).astype(int).values[0]
    total_score = mcq_score + essay_score+code_score
    
    
     
    # Pie chart data: Only MCQ, Essay, and Code Test
    categories = ['MCQ', 'Essay', 'Code Test']
    scores = [mcq_score, essay_score, code_score]
    
    # Generate Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    
    wedges, texts, autotexts = ax.pie(
        scores, 
        labels=categories, 
        autopct='%1.1f%%',
        colors=['blue', 'orange', 'green'],  
        startangle=90,
        wedgeprops={'edgecolor': 'black'}
    )
    
    # Add total score label outside the pie chart
    plt.text(1, 1, f"Total Score: {total_score}", fontsize=12, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_title(f'Score Distribution for {reg_no}')
    
    # Save Pie Chart
    plot_path = f"static/results/score_pie_{reg_no}.png"
    plt.savefig(plot_path, bbox_inches='tight')  # Save image with all elements
    plt.close()
    feedback_csv = f'static/results/feedback_{reg_no}.csv'
   # Read CSV into DataFrame
    
    if os.path.exists(feedback_csv):
        # Load feedback if essay feedback exists
        df = pd.read_csv(feedback_csv)
        feedback_html = df.to_html(index=False, escape=False)
    else:
        # Placeholder message if essay not taken yet
        feedback_html = "<p>No essay feedback available yet.</p>"


    #   Convert to HTML string (no saving to file)
    #feedback_html = df.to_html(index=False, escape=False)
    return render_template('result1.html', mcq_score=mcq_score, essay_score=essay_score,code_score=code_score, total_score=total_score, feedback_csv=feedback_csv,feedback_table=feedback_html,plot_path=plot_path)

@app.route('/student_results')
def student_results():
    if 'username' not in session:
        return redirect(url_for('student_login'))
    return redirect(url_for('view_results'))


@app.route('/grade_essay/<filename>')
def grade_essay(filename):
    if 'username' not in session:
        return redirect(url_for('student_login'))

    reg_no = session['username']
    responses_df = pd.read_csv(os.path.join('uploads', filename))

    student_responses = responses_df.set_index('Question')['Response'].to_dict()
    final_scores = []
    feedbacks = []
    question_feedback = []

    for idx, row in questions_df_essay.iterrows():
        question = row['Question']
        model_answer = row['Answer']
        student_answer = student_responses.get(question, "")
        
        if not student_answer:
            final_scores.append(0.0)
            feedbacks.append("No response provided.")
            question_feedback.append({"Question": question, "Score": 0.0, "Feedback": "No response provided."})
            continue

        # Calculate sentence transformer similarity
        transformer_scores = compute_transformer_similarity([model_answer], student_answer)
        transformer_score = round(transformer_scores[0] * 10, 2)

        final_scores.append(transformer_score)

        # Feedback Mechanism
        if transformer_score >= 8:
            feedback = "Excellent response! Strong understanding."
        elif transformer_score >= 6:
            feedback = "Good response. Minor improvements needed."
        elif transformer_score >= 4:
            feedback = "Fair response. Needs more development."
        else:
            feedback = "Needs significant improvement. Focus on key concepts."

        feedbacks.append(feedback)
        question_feedback.append({"Question": question, "Score": transformer_score, "Feedback": feedback})

    essay_score = sum(final_scores)
    store_scores(reg_no, essay_score=essay_score)

    # Save feedback and scores
    feedback_df = pd.DataFrame(question_feedback)
    feedback_csv_path = f"static/results/feedback_{reg_no}.csv"
    feedback_df.to_csv(feedback_csv_path, index=False)
    feedback_html = feedback_df.to_html(index=False, classes='table table-striped')
    
    # Load MCQ score
    scores_df = pd.read_csv("score_df.csv")
    mcq_score = scores_df.loc[scores_df["Reg.No."] == reg_no, "MCQ Score"].values[0]
    code_score = scores_df.loc[scores_df["Reg.No."] == reg_no, "Code Score"].values[0]
    total_score = mcq_score + essay_score + code_score

   
    feedback_csv = f'/static/results/feedback_{reg_no}.csv'

    return render_template(
        'result11.html',
        essay_score=essay_score,
        #mcq_score=mcq_score,
        #code_score=code_score,
        #total_score=total_score,
        feedback_table=feedback_html,
        feedback_csv=feedback_csv,
        #plot_path=plot_path,
        reg_no=reg_no
    )

if __name__ == '__main__':
    app.run(debug=True)