# Essay Scoring Application

This is a Flask-based web application for scoring essay responses. The application uses keyword matching, TF-IDF and BERT-based similarity measures to evaluate the responses and provide feedback.

## Features

- User authentication
- Upload essay responses
- Automatic grading of responses using keyword matching,TF-IDF and BERT-based similarity
- Detailed feedback for each response
- Summary of total score

## Requirements

- Python 3.9
- Flask
- pandas
- nltk
- numpy
- scikit-learn
- transformers
- torch
- 
# Objective Quiz Application

This is a Flask-based web application for conducting multiple-choice quizzes. The application allows users to log in, take a quiz, and receive their scores.

## Features

- User authentication
- Multiple-choice quiz
- Automatic grading of quiz responses
- Score storage

## Requirements

- Python 3.9
- Flask
- pandas
# COMBINED APP FOR ESSAY AND MCQ SCORE

## Features

### Student Dashboard
- **MCQ Test**: Students can attend multiple-choice questions.
- **Essay Test**: Students can submit their essay responses.
- **View Results**: Students can view their results, including scores for both MCQ and essay tests.

### Faculty Dashboard
- **View Student Results**: Faculty can view the results of all students who have attended the exam.
- **View Plots**: Faculty can view bar charts showing total scores for all students.

### Essay Grading
- **BERT Transformer Model**: The essay grading logic uses the **BERT transformer model** to evaluate essays, significantly improving accuracy compared to previous methods like **keyword matching** and **TF-IDF cosine similarity**.

### Download Options
- **PDF Download**: Both students and faculty can download the result page as a **PDF**.
- **CSV Download**: Results are also available for download in **CSV** format.

---

### Additional Notes
This app is part of the **FYUGP Smart Grading System**, combining **MCQ and Essay Grading** into a single interface, with features for both students and faculty.




