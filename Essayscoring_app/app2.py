from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Load Hugging Face Transformer Model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize Flask App
app = Flask(__name__)
app.secret_key = '123545'

# Lemmatizer for better text normalization
lemmatizer = WordNetLemmatizer()

# Function to preprocess text with lemmatization
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return set(tokens)  # Using a set for faster keyword checking

# Function to compute TF-IDF similarity
def compute_tfidf_similarity(model_answers, student_answer):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(model_answers + [student_answer])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    return similarities

# Function to compute BERT-based similarity using Hugging Face
def compute_bert_similarity(model_answers, student_answer):
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

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        reg_no = request.form['reg_no']
        password = request.form['password']
        users_df = pd.read_csv('users.csv')
        user = users_df[(users_df['Reg.No.'] == reg_no) & (users_df['Password'] == password)]
        if not user.empty:
            session['reg_no'] = reg_no
            return redirect(url_for('upload_response'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_response():
    if 'reg_no' not in session:
        return redirect(url_for('login'))
    
    # Load questions from CSV
    questions_df = pd.read_csv('essayquestion.csv')
    questions = questions_df['Question'].tolist()  # Only get the questions
    
    if request.method == 'POST':
        file = request.files['response_file']
        if file:
            file.save(os.path.join('uploads', file.filename))
            return redirect(url_for('grade', filename=file.filename))
    
    return render_template('upload.html', questions=questions)

@app.route('/grade/<filename>')
def grade(filename):
    if 'reg_no' not in session:
        return redirect(url_for('login'))

    reg_no = session['reg_no']  
    questions_df = pd.read_csv('essayquestion.csv')
    responses_df = pd.read_csv(os.path.join('uploads', filename))

    student_responses = responses_df.set_index('Question')['Response'].to_dict()
    final_scores = []
    feedbacks = []

    for idx, row in questions_df.iterrows():
        question = row['Question']
        model_answer = row['Answer']
        student_answer = student_responses.get(question, "")

        if not student_answer:
            final_scores.append(0.0)
            feedbacks.append("No response provided.")
            continue

        student_words = preprocess_text(student_answer)
        tfidf_scores = compute_tfidf_similarity([model_answer], student_answer)
        bert_scores = compute_bert_similarity([model_answer], student_answer)

        keywords = row['Keywords'].split(',') if pd.notna(row['Keywords']) else []
        keywords = [word.strip().lower() for word in keywords]

        matched_keywords = sum(1 for word in keywords if word in student_words)
        keyword_score = (matched_keywords / len(keywords)) * 10 if keywords else 0.0
        tfidf_score = round(tfidf_scores[0] * 10, 2)
        bert_score = round(bert_scores[0] * 10, 2)

        final_score = round((0.4 * keyword_score) + (0.3 * tfidf_score) + (0.3 * bert_score), 2)
        final_scores.append(final_score)

        # 
                # Enhanced Feedback Mechanism
        if final_score >= 8:
            feedback = "Excellent response! Demonstrates a strong understanding of the concept."
        elif final_score >= 6:
            feedback = "Good response. Minor improvements could enhance clarity and depth."
        elif final_score >= 4:
            feedback = "Fair response. Needs further development and elaboration on key points."
            missing_keywords = [kw for kw in keywords if kw not in student_words]
            if missing_keywords:
                feedback += f" Consider including these keywords: {', '.join(missing_keywords)}."
        else:
            feedback = "Response needs significant improvement. Focus on addressing the core concepts of the question."
            if keywords:
                feedback += f" Make sure to include key terms like: {', '.join(keywords)}."
        feedbacks.append(feedback)


    questions_df['score'] = final_scores
    questions_df['Feedback'] = feedbacks
    questions_df['Reg.No.'] = reg_no
    total_score = questions_df['score'].sum()

    score_df = pd.DataFrame({'Reg.No': [reg_no], 'Total_Score': [total_score]})
    score_df.to_csv('score_df.csv', index=False)

    detailed_feedback_df = questions_df[['Reg.No.','Question', 'score', 'Feedback']]
    detailed_feedback_df.to_csv('detailed_feedback.csv', index=False)

    return render_template('result.html', questions=questions_df.to_dict(orient='records'), total_score=total_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
