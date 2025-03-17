import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pickle
import os

# Function to train the model and generate predictions
def train_and_generate_results():
    """Trains the model, predicts scores, and saves relevant CSV files."""
    user = pd.read_csv('user_data.csv')
    print(user.head()) # Debugging

    # Select relevant columns
    user_df = user[['Age', 'Gender', '10th %', '12th %', 'Stream', 'Total Score']]

    # Encode categorical values
    distinct_streams = user_df['Stream'].unique()
    user_df['Gender'] = user_df['Gender'].map({'Male': 0, 'Female': 1})
    stream_mapping = {stream: i for i, stream in enumerate(distinct_streams)}
    inverse_stream_mapping = {i: stream for stream, i in stream_mapping.items()}  # Fix

    user_df['Stream'] = user_df['Stream'].map(stream_mapping)

    # Split data
    x = user_df.drop('Total Score', axis=1)
    y = user_df['Total Score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # Train model
    model = LGBMRegressor(random_state=42, learning_rate=0.1)
    model.fit(x_train, y_train)

    # Save model & stream mapping
    #pickle.dump(model, open('model.pkl', 'wb'))
    #pickle.dump(inverse_stream_mapping, open('stream_mapping.pkl', 'wb'))  # Fix

    # Predict total scores
    user_df["Predicted Total Score"] = model.predict(x)

    # Calculate stream-wise performance
    stream_performance = user_df.groupby("Stream").agg(
        Actual_Avg_Score=("Total Score", "mean"),
        Predicted_Avg_Score=("Predicted Total Score", "mean"),
        Student_Count=("Stream", "count")
    ).reset_index()

    # Calculate standard deviation and intervention threshold
    std_dev = user_df["Total Score"].std()
    overall_predicted_avg = user_df["Predicted Total Score"].mean()
    overall_actual_avg = user_df["Total Score"].mean()
    urgent_intervention_threshold = overall_predicted_avg - (1.5 * std_dev)  # Fix: Dynamic threshold

    # Assign performance feedback
    def performance_feedback(row):
        if row["Predicted_Avg_Score"] >= overall_predicted_avg:
            return "Good Performance"
        elif row["Predicted_Avg_Score"] >= urgent_intervention_threshold:
            return "Average, Needs Improvement"
        else:
            return "Needs Significant Improvement (Urgent Intervention)"

    stream_performance["Performance_Feedback"] = stream_performance.apply(performance_feedback, axis=1)

    # Save results
    #user_df.to_csv('user_data_with_predictions.csv', index=False)
    stream_performance.to_csv('stream_performance.csv', index=False)

    return stream_performance, overall_actual_avg, overall_predicted_avg

# Function to generate and save the streamwise performance plot
def generate_streamwise_plot():
    """Generates and saves the streamwise performance plot."""
    if not os.path.exists("stream_performance.csv"):
        print("Error: stream_performance.csv not found!")
        return None

    stream_performance = pd.read_csv("stream_performance.csv")
    inverse_stream_mapping = pickle.load(open('stream_mapping.pkl', 'rb'))  # Load mapping

    plt.figure(figsize=(10, 6))
    plt.plot(stream_performance["Stream"], stream_performance["Actual_Avg_Score"], marker='o', label="Actual Avg Score")
    plt.plot(stream_performance["Stream"], stream_performance["Predicted_Avg_Score"], marker='x', label="Predicted Avg Score")

    plt.xticks(stream_performance["Stream"], [inverse_stream_mapping[stream] for stream in stream_performance["Stream"]], rotation=45, ha="right")
    plt.xlabel("Stream")
    plt.ylabel("Average Total Score")
    plt.title("Actual vs. Predicted Average Scores per Stream")
    plt.legend()
    plt.grid(True)

    os.makedirs("static/results", exist_ok=True)
    plot_path = "static/results/stream_performance_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# Function to generate and save the bar plot for stream-wise actual vs predicted performance
def generate_bar_plot():
    """Generates and saves the bar plot comparing actual vs predicted scores for each stream."""
    if not os.path.exists("stream_performance.csv"):
        print("Error: stream_performance.csv not found!")
        return None

    stream_performance = pd.read_csv("stream_performance.csv")
    inverse_stream_mapping = pickle.load(open('stream_mapping.pkl', 'rb'))  # Load mapping

    overall_actual_avg = stream_performance["Actual_Avg_Score"].mean()
    overall_predicted_avg = stream_performance["Predicted_Avg_Score"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(stream_performance["Stream"], stream_performance["Actual_Avg_Score"], label="Actual Avg Score", alpha=0.7)
    plt.bar(stream_performance["Stream"], stream_performance["Predicted_Avg_Score"], label="Predicted Avg Score", alpha=0.7)
    plt.axhline(y=overall_actual_avg, color="r", linestyle="--", label="Overall Actual Avg")
    plt.axhline(y=overall_predicted_avg, color="g", linestyle="--", label="Overall Predicted Avg")
    inverse_stream_mapping = pickle.load(open("stream_mapping.pkl", "rb"))

    plt.xticks(
    stream_performance["Stream"],
    [inverse_stream_mapping.get(stream, "Unknown") for stream in stream_performance["Stream"]],
    rotation=45, ha="right"
    )
    #plt.xticks(stream_performance["Stream"], [inverse_stream_mapping[stream] for stream in stream_performance["Stream"]], rotation=45, ha="right")
    plt.xlabel("Stream")
    plt.ylabel("Total Score")
    plt.title("Stream-wise Actual vs Predicted Performance")
    plt.legend()

    os.makedirs("static/results", exist_ok=True)
    plot_path = "static/results/stream_performance_bar_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# Function to evaluate student performance

def evaluate_student_performance(student_data, model, stream_avg_scores, overall_avg_score, stream_mapping):
    """Evaluates an individual student's performance using the trained model."""
    model = pickle.load(open('model.pkl', 'rb'))  # Load model
    # Convert categorical "Stream" to its numeric value
    stream_name = student_data.get("Stream", "").strip()  # Ensure it's a string
    stream_numeric = stream_mapping.get(stream_name, -1)  # Default to -1 if stream is not found

    # Define feature names (Must match training)
    feature_names = ["Age", "Gender", "10th %", "12th %", "Stream"]

    # Ensure all inputs are properly converted to numeric
    student_features = pd.DataFrame([[
        pd.to_numeric(student_data.get("Age", 0), errors="coerce"),
        0 if str(student_data.get("Gender", "")).strip().lower() == "male" else 1,  # Convert gender
        pd.to_numeric(student_data.get("10th %", "0"), errors="coerce"),
        pd.to_numeric(student_data.get("12th %", "0"), errors="coerce"),
        stream_numeric  # Use the mapped numeric stream value
    ]], columns=feature_names)  # ✅ Ensure feature names are present

    # Ensure no NaN values in features
    if student_features.isnull().values.any():
        return {"error": "Error: Missing or invalid data in student features!"}

    # Ensure student_features is not empty before passing to the model
    if student_features.empty:
        return {"error": "Error: Empty feature array!"}

    # Predict total score
    predicted_score = model.predict(student_features)[0]  # ✅ Now model gets named features

    # Get stream’s predicted average score safely
    stream_avg = stream_avg_scores.loc[stream_avg_scores["Stream"] == stream_numeric, "Predicted_Avg_Score"].values
    stream_avg_score = stream_avg[0] if stream_avg.size > 0 else overall_avg_score  # ✅ Corrected check

    # Ensure Total Score is retrieved safely
    total_score = student_data.get("Total Score", 0)  # Default to 0 if missing

    # Stream Performance Feedback
    if total_score >= stream_avg_score:
        stream_feedback = "Good Performance"
    elif total_score >= stream_avg_score - 10:
        stream_feedback = "Average Performance"
    else:
        stream_feedback = "Needs Improvement in Stream"

    # Overall Performance Feedback
    if total_score >= overall_avg_score:
        overall_feedback = "Good Overall Performance"
    elif total_score >= overall_avg_score - 10:
        overall_feedback = "Average Overall Performance"
    else:
        overall_feedback = "Needs Improvement Overall"

    return {
        "Actual Score": total_score,
        "Predicted Score": predicted_score,
        "Stream Avg Predicted Score": stream_avg_score,
        "Overall Avg Predicted Score": overall_avg_score,
        "Stream Performance Feedback": stream_feedback,
        "Overall Performance Feedback": overall_feedback
    }

# Function to generate the student_performance bar chart
def generate_student_performance_bar_chart(actual_score, predicted_score, student_stream_avg, overall_avg_score, student_id):
    """
    Generates a bar chart comparing the student's actual score with predicted score, stream average, and overall average.
    
    Parameters:
    - actual_score: The student's actual score.
    - predicted_score: The predicted score by the model.
    - student_stream_avg: The predicted stream average.
    - overall_avg_score: The overall average predicted score.
    - student_id: The student's ID or reg number for saving the plot.
    
    Returns:
    - The path to the saved bar chart image.
    """
    plt.figure(figsize=(8, 5))
    categories = ['Actual Score', 'Predicted Score', 'Stream Avg', 'Overall Avg']
    values = [actual_score, predicted_score, student_stream_avg, overall_avg_score]

    plt.bar(categories, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 200)
    plt.ylabel("Total Score")
    plt.title(f"Student {student_id} Performance Comparison")
    plt.axhline(y=student_stream_avg, color='orange', linestyle='dashed', label="Stream Avg")
    plt.axhline(y=overall_avg_score, color='red', linestyle='dashed', label="Overall Avg")
    plt.legend()

    # Save the plot to the 'static/results' folder
    plot_path = f'static/results/student_{student_id}_performance_comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path
if __name__ == "__main__":
    print("Starting model training...")
    stream_performance, overall_actual_avg, overall_predicted_avg = train_and_generate_results()
    print("Model training completed. Model should be saved as 'model.pkl'.")
    print("Stream performance data saved as 'stream_performance.csv'.")
        # Load student data and stream performance data
    model= pickle.load(open('model.pkl', 'rb'))
    stream_mapping = pickle.load(open('stream_mapping.pkl', 'rb'))
    student_df = pd.read_csv('user_data.csv')
    stream_avg_scores = pd.read_csv('stream_performance.csv')
    overall_avg_score = stream_avg_scores['Predicted_Avg_Score'].mean()

    student_performance_list = []

        # Loop through each student and evaluate performance
    for _, student_data in student_df.iterrows():
            student_dict = student_data.to_dict()
            performance_feedback = evaluate_student_performance(student_dict, model, stream_avg_scores, overall_avg_score,stream_mapping)

            # Add student details and performance to list
            student_performance_list.append({
                "Reg.No.": student_dict["Reg.No."],
                "Name": student_dict["Name"],
                "Stream": student_dict["Stream"],
                "Age": student_dict["Age"],
                "Gender": student_dict["Gender"],
                "Total Score": performance_feedback["Actual Score"],
                "Predicted Score": performance_feedback["Predicted Score"],
                "Stream Avg": performance_feedback["Stream Avg Predicted Score"],
                "Overall Avg": performance_feedback["Overall Avg Predicted Score"],
                "Stream Feedback": performance_feedback["Stream Performance Feedback"],
                "Overall Feedback": performance_feedback["Overall Performance Feedback"]
            })
        #Convert list to DataFrame
    performance_df = pd.DataFrame(student_performance_list)

        # Save as CSV file
    csv_filename = "static/results/student_performance.csv"
    performance_df.to_csv(csv_filename, index=False)

    print(f"Student performance evaluation completed. Results saved as '{csv_filename}'.")
   
