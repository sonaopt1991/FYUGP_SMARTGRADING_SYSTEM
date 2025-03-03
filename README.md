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
## combined mcq and essay scoring app
## overview 
--student dashboard is added ,where student can choose from mcq,essay test options,they can view there result after attending the test.plot is also included and an option to export the result page as pdf.
---faculty dashboard is included where faculty can view the results of all student in table format and option to download it as pdf.
---option to view result as barchart .
-----for student essay evaluation used BERT TRANSFORMER model 
## Requirements

- Python 3.9
- Flask
- pandas
- nltk
- numpy
- scikit-learn
- transformers
- torch
- matplotlib
- python blueprint 




## code_app
## Overview
Gemini 1.5 App is a web application that provides AI-generated feedback on students' code submissions. It uses Google Gemini 1.5 Pro to evaluate the correctness, efficiency, readability, and possible improvements of the code. The application is designed to assist both students and faculty by providing structured feedback and overall scores for code submissions.
- **AI Feedback**: Utilizes Google Gemini 1.5 Pro to generate detailed feedback on code correctness, efficiency, readability, and improvements.
- **Overall Score**: Provides an overall score out of 10 for each code submission.
  ## Requirements

  --dotenv
  --gemini1.5 api key
  
