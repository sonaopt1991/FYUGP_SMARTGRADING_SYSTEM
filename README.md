## code_app
## Overview
Gemini 1.5 App is a web application that provides AI-generated feedback on students' code submissions. It uses Google Gemini 1.5 Pro to evaluate the correctness, efficiency, readability, and possible improvements of the code. The application is designed to assist both students and faculty by providing structured feedback and overall scores for code submissions.
- **AI Feedback**: Utilizes Google Gemini 1.5 Pro to generate detailed feedback on code correctness, efficiency, readability, and improvements.
- **Overall Score**: Provides an overall score out of 10 for each code submission.
## code_app gemma
## overview 
Developed an AI based web application using python flask framework to evaluate student code response and give feedback .
- Model used: pulled "gemma:2b" model through ollama.used this gemma model for evaluating student's code ,which is submitted as a csv with 'Question' column and 'Response' column. Model evaluates this based on
- Code correctness
- Efficiency(BIG O notation)
- Readability
- alternative improvements are provided
- assign score based on these  masures .
