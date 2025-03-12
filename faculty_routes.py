# # faculty_routes.py
#from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask import Blueprint, render_template, session, redirect, url_for, flash,request, send_file,session
import pandas as pd
import os
import matplotlib.pyplot as plt

faculty_bp = Blueprint('faculty', __name__)

# Faculty Login
@faculty_bp.route('/faculty_login', methods=['GET', 'POST'])
def faculty_login():
    faculty_credentials_df = pd.read_csv("FACULTY.csv")
    faculty_credentials = dict(zip(faculty_credentials_df['username'], faculty_credentials_df['password']))

    if request.method == 'POST':
        faculty_id = request.form['username']
        password = request.form['password']

        if faculty_id in faculty_credentials and faculty_credentials[faculty_id] == password:
            session['username'] = faculty_id
            session['role'] = 'faculty'
            return redirect(url_for('faculty.faculty_dashboard'))
        else:
            flash("Invalid faculty credentials! Try again.")
    
    return render_template('faculty_login.html')

# Faculty Dashboard
@faculty_bp.route('/faculty_dashboard')
def faculty_dashboard():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty.faculty_login'))

    return render_template('faculty_dashboard.html')

# View Results (Table)
@faculty_bp.route('/faculty_view_results')
def faculty_view_results():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty.faculty_login'))

    if os.path.exists("score_df.csv"):
        scores_df = pd.read_csv("score_df.csv")
    else:
        scores_df = pd.DataFrame(columns=["Reg.No.", "MCQ Score", "Essay Score"])

    return render_template('faculty_result.html', scores_df=scores_df)

# View Plots (Bar Chart of Total Scores)
@faculty_bp.route('/faculty_view_plots')
def faculty_view_plots():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty.faculty_login'))

    if os.path.exists("score_df.csv"):
        scores_df = pd.read_csv("score_df.csv")
        scores_df['Total Score'] = scores_df['MCQ Score'] + scores_df['Essay Score']

        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.bar(scores_df['Reg.No.'], scores_df['Total Score'], color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel('Total Score')
        plt.title('Total Scores of Students')

        os.makedirs('static/results', exist_ok=True)
        plot_path = 'static/results/faculty_total_scores_plot.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    else:
        plot_path = None

    return render_template('faculty_plots.html', plot_path=plot_path)

# Faculty Logout
@faculty_bp.route('/faculty_logout')
def faculty_logout():
    session.pop('username', None)
    flash("Faculty logged out successfully.")
    #return redirect(url_for('faculty.faculty_login'))
    #return redirect(url_for('faculty.landing_page'))
    return redirect(url_for('home'))
