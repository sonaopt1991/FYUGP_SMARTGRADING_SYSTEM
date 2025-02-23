from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load data
credentials_df = pd.read_csv("user.csv")
questions_df = pd.read_csv("MCQ_QUESTION.csv")

def load_credentials():
    return dict(zip(credentials_df['Reg.No.'], credentials_df['Password']))

# MCQ Grader class
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

# Store user scores
def store_mcq_score(username, score, score_file="mcq_scores.csv"):
    try:
        scores_df = pd.read_csv(score_file)
    except FileNotFoundError:
        scores_df = pd.DataFrame(columns=["Reg.No.", "MCQ Score"])

    if username in scores_df["Reg.No."].values:
        scores_df.loc[scores_df["Reg.No."] == username, "MCQ Score"] = score
    else:
        new_row = pd.DataFrame({"Reg.No.": [username], "MCQ Score": [score]})
        scores_df = pd.concat([scores_df, new_row], ignore_index=True)

    scores_df.to_csv(score_file, index=False)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        credentials = load_credentials()
        
        if username in credentials and credentials[username] == password:
            session['username'] = username
            return redirect(url_for('quiz'))
        else:
            return "Invalid credentials! Try again."
    return render_template('login.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    grader = Grader(questions_df)
    questions = grader.load_questions()
    options = grader.load_options()
    
    if request.method == 'POST':
        user_answers = {q: request.form[q] for q in questions}
        score, total_questions = grader.grade_answers(user_answers)
        store_mcq_score(session['username'], score)
        return f"Your Score: {score}/{total_questions}"

    return render_template('quiz.html', questions=questions, options=options)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
