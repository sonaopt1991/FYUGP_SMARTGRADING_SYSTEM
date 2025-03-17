from flask import Blueprint, render_template, session, redirect, url_for, flash,request, send_file
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import train_and_generate_results, generate_streamwise_plot,evaluate_student_performance,generate_bar_plot,generate_student_performance_bar_chart
from plots import generate_top_students_bar_chart, generate_student_distribution_box_plot, generate_student_histogram
import pickle
# Load the model (make sure it's available when the app starts)
model = pickle.load(open('model.pkl', 'rb'))
faculty1_bp = Blueprint('faculty1', __name__)

# Faculty Login
@faculty1_bp.route('/faculty_login', methods=['GET', 'POST'])
def faculty_login():
    faculty_credentials_df = pd.read_csv("FACULTY.csv",dtype=str, encoding='utf-8')
    faculty_credentials = dict(zip(faculty_credentials_df['username'].str.strip(), faculty_credentials_df['password'].astype(str).str.strip()))

    if request.method == 'POST':
        # faculty_id = request.form['username']
        # password = request.form['password']
        faculty_id = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if faculty_id in faculty_credentials and faculty_credentials[faculty_id] == password:
            session['username'] = faculty_id
            session['role'] = 'faculty'
            return redirect(url_for('faculty1.faculty_dashboard'))
        else:
            flash("Invalid faculty credentials! Try again.")
    
    return render_template('faculty_login.html')

# Faculty Dashboard
@faculty1_bp.route('/faculty_dashboard')
def faculty_dashboard():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty1.faculty_login'))

    return render_template('faculty_dashboard1.html')

# 📊 Faculty View All Students Performance

# Faculty View All Students Performance
@faculty1_bp.route('/faculty_view_all_students_performance')
def faculty_view_all_students_performance():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty1.faculty_login'))

    # Load student performance data
    file_path = "static\\results\\student_performance.csv"
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading student performance data: {str(e)}"

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 150  # Number of rows per page
    total_rows = len(df)
    total_pages = (total_rows // per_page) + (1 if total_rows % per_page > 0 else 0)

    # Get paginated data
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    df_paginated = df.iloc[start_idx:end_idx]

    # Convert DataFrame to HTML table
    table_html = df_paginated.to_html(classes='table table-striped', index=False)

    return render_template(
        "faculty_view_students.html",
        table_html=table_html,
        page=page,
        total_pages=total_pages
    )


# 📊 Faculty View Plots Route
@faculty1_bp.route('/faculty_view_plots')
def faculty_view_plots():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty1.faculty_login'))

    try:
        # # Generate plots and store paths
        # top_students_chart = generate_top_students_bar_chart(model)
        # box_plot_chart = generate_student_distribution_box_plot()
        # histogram_chart = generate_student_histogram()

        return render_template('faculty_plots1.html', top_students_chart=url_for('static', filename='results/top_students_bar_chart.png'),
        box_plot_chart=url_for('static', filename='results/student_distribution_boxplot.png'),
        histogram_chart=url_for('static', filename='results/student_score_histogram.png'))

    except Exception as e:
        flash(f"Error generating plots: {str(e)}", "error")
        return redirect(url_for('faculty1.faculty_dashboard'))
    
# Streamwise Performance
@faculty1_bp.route('/streamwise_performance')
def streamwise_performance():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty1.faculty_login'))

    # Ensure data is available
    stream_performance, _, _ = train_and_generate_results()

    if stream_performance is None:
        flash("Stream performance data not found!", "error")
        return redirect(url_for('faculty1.faculty_dashboard'))

    # Generate both plots
    line_plot_path = generate_streamwise_plot()
    bar_plot_path = generate_bar_plot()

    # Load stream_performance.csv into a DataFrame
    stream_df = pd.read_csv("stream_performance.csv")

    return render_template("streamwise_performance.html", 
                           line_plot=url_for('static', filename='results/stream_performance_plot.png'),
                           bar_plot=url_for('static', filename='results/stream_performance_bar_plot.png'),
                           stream_data=stream_df.to_dict(orient='records'))  # Convert DataFrame to list of dicts


# # View Specific Student's Performance
# @faculty1_bp.route('/faculty_view_student_performance', methods=['GET', 'POST'])


# def faculty_view_student_performance():
#     if 'username' not in session or session.get('role') != 'faculty':
#         return redirect(url_for('faculty1.faculty_login'))
    
#     print("Faculty viewing all students' performance")
#     # Load stream performance data and calculate overall average score
#     stream_avg_scores = pd.read_csv('stream_performance.csv')
    
#     overall_avg_score = stream_avg_scores['Predicted_Avg_Score'].mean()  # Example of overall average score
    
#     if request.method == 'POST':
#         reg_no = request.form['reg_no']
#         try:
#         # Assuming we have a CSV file with student details like user_data.csv
#             student_df = pd.read_csv('user_data.csv')
#             print("Student data loaded successfully")
#              # Ensure Reg.No. exists before accessing data
#             if reg_no not in student_df['Reg.No.'].values:
#                 flash(f"Error: Student with Reg.No. {reg_no} not found!", "error")
#                 return redirect(url_for('faculty1.faculty_view_student_performance'))
        
#         # Fetch student details using reg_no
#             student_data = student_df[student_df['Reg.No.'] == reg_no].iloc[0].to_dict()
        
#         # Use the evaluate_student_performance function to get feedback
#             performance_feedback = evaluate_student_performance(student_data, model, stream_avg_scores, overall_avg_score)
#             #performance_feedback = evaluate_student_performance(student_data, model, Stream_performance, overall_avg_score)
#         # Generate student performance bar chart
#             generate_student_performance_bar_chart(
#             actual_score=performance_feedback["Actual Score"],
#             predicted_score=performance_feedback["Predicted Score"],
#             student_stream_avg=performance_feedback["Stream Avg Predicted Score"],
#             overall_avg_score=performance_feedback["Overall Avg Predicted Score"],
#             student_id=reg_no)
#             return render_template('faculty_student_performance.html', 
#                                reg_no=reg_no,
#                                performance_feedback=performance_feedback)
#         except FileNotFoundError:
#             flash("Error: The required student data file (`user_data.csv`) is missing!", "error")
#             return redirect(url_for('faculty1.faculty_dashboard'))
        
#         except pd.errors.EmptyDataError:
#             flash("Error: `user_data.csv` is empty!", "error")
#             return redirect(url_for('faculty1.faculty_dashboard'))
        
#         except Exception as e:
#             flash(f"An unexpected error occurred: {str(e)}", "error")
#             return redirect(url_for('faculty1.faculty_dashboard'))
    
#     return render_template('faculty_view_student_performance.html')
# # View Specific Student's Performance
@faculty1_bp.route('/faculty_view_student_performance', methods=['GET', 'POST'])
def faculty_view_student_performance():
    if 'username' not in session or session.get('role') != 'faculty':
        return redirect(url_for('faculty1.faculty_login'))
    
    if request.method == 'POST':
        reg_no = request.form['reg_no']
        
        try:
            # Load student performance data
            student_df = pd.read_csv('student_performance (2).csv')

            # Check if the given registration number exists
            if reg_no not in student_df['Reg.No.'].values:
                flash(f"Error: Student with Reg.No. {reg_no} not found!", "error")
                return redirect(url_for('faculty1.faculty_view_student_performance'))
            
            # Fetch student details using reg_no
            student_data = student_df[student_df['Reg.No.'] == reg_no].iloc[0]

            # Extract required values directly from the CSV
            actual_score = student_data["Total Score"]
            predicted_score = student_data["Predicted Score"]
            student_stream_avg_score = student_data["Stream Avg"]
            overall_avg_score = student_data["Overall Avg"]
            plt.figure(figsize=(8, 5))
            categories = ['Actual Score', 'Predicted Score', 'Stream Avg', 'Overall Avg']
            values = [actual_score, predicted_score, student_stream_avg_score, overall_avg_score]

            plt.bar(categories, values, color=['blue', 'green', 'orange', 'red'])
            plt.ylim(0, 200)
            plt.ylabel("Total Score")
            plt.title(f"Student {reg_no} Performance Comparison")
            plt.axhline(y=student_stream_avg_score, color='orange', linestyle='dashed', label="Stream Avg")
            plt.axhline(y=overall_avg_score, color='red', linestyle='dashed', label="Overall Avg")
            plt.legend()

            # Save the plot to the 'static/results' folder
            plot_path = f'static/results/student_{reg_no}_performance_chart.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()


            # # Get bar chart path (assuming it’s pre-generated and saved)
            # plot_path = f"static/results/{reg_no}_performance_chart.png"

            return render_template('faculty_view_student_performance.html', 
                                   reg_no=reg_no,
                                   actual_score=actual_score,
                                   predicted_score=predicted_score,
                                   student_stream_avg_score=student_stream_avg_score,
                                   overall_avg_score=overall_avg_score,
                                   plot_path=plot_path)
        
        except FileNotFoundError:
            flash("Error: `student_performance.csv` is missing!", "error")
        except pd.errors.EmptyDataError:
            flash("Error: `student_performance.csv` is empty!", "error")
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "error")

        return redirect(url_for('faculty1.faculty_dashboard'))

    return render_template('faculty_view_student_performance.html')

# Faculty Logout
@faculty1_bp.route('/faculty_logout')
def faculty_logout():
    session.pop('username', None)
    flash("Faculty logged out successfully.")
    return redirect(url_for('home'))
