import os
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ FIX: Prevent Matplotlib GUI errors
matplotlib.use('Agg')

# ✅ Load trained model
model = pickle.load(open('model.pkl', 'rb'))


# 📊 1️⃣ **Grouped Bar Chart (Top 10 Students)**
def generate_top_students_bar_chart(model):
    student_df = pd.read_csv('student_performance (2).csv')

    # ✅ Load stream mapping
    with open("stream_mapping.pkl", "rb") as f:
        stream_mapping = pickle.load(f)

    # ✅ Encode Gender & Stream
    gender_mapping = {'Male': 0, 'Female': 1}
    student_df['Gender'] = student_df['Gender'].map(gender_mapping)
    student_df['Stream'] = student_df['Stream'].map(stream_mapping)

    # ✅ Ensure numeric conversion and drop missing values
    student_df[['Gender', 'Stream']] = student_df[['Gender', 'Stream']].apply(pd.to_numeric, errors='coerce')
    student_df = student_df.dropna()

    # ✅ Sort by actual score & take top 10 students
    top_students = student_df.sort_values(by="Total Score", ascending=False).head(10)

    # ✅ Extract features & reshape to ensure 2D format
    features = top_students[['Age', 'Gender', '10th %', '12th %', 'Stream']].values
    features = features.reshape(-1, 5)  

    # ✅ Predict scores
    predicted_scores = model.predict(features)

    # ✅ Plot: Actual vs Predicted Scores
    plt.figure(figsize=(12, 6))
    x = top_students["Reg.No."]
    actual_scores = top_students["Total Score"]

    plt.bar(x, actual_scores, color='blue', width=0.4, label="Actual Score", align='center')
    plt.bar(x, predicted_scores, color='orange', width=0.4, label="Predicted Score", align='edge')

    plt.xlabel("Reg. No.")
    plt.ylabel("Total Score")
    plt.title("Top 10 Students: Actual vs Predicted Performance")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # ✅ Save plot
    plot_path = "static/results/top_students_bar_chart.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# 📦 2️⃣ **Box Plot (Score Distribution)**
def generate_student_distribution_box_plot():
    student_df = pd.read_csv('student_performance (2).csv')

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=student_df["Total Score"], color='skyblue')

    plt.ylabel("Total Score")
    plt.title("Student Score Distribution")

    # ✅ Save plot
    plot_path = "static/results/student_distribution_boxplot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# 📉 3️⃣ **Histogram (Frequency of Score Ranges)**
def generate_student_histogram():
    student_df = pd.read_csv('student_performance (2).csv')

    plt.figure(figsize=(10, 6))
    plt.hist(student_df["Total Score"], bins=20, color='blue', alpha=0.7, edgecolor='black')

    plt.xlabel("Total Score Range")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Student Scores")

    # ✅ Save plot
    plot_path = "static/results/student_score_histogram.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# 🔹 **Run Script: Generate & Save Plots**
if __name__ == "__main__":
    # ✅ Ensure the results directory exists
    os.makedirs("static/results", exist_ok=True)

    print("📊 Generating and saving plots...")

    # ✅ Call each function to generate and save plots
    generate_top_students_bar_chart(model)
    generate_student_distribution_box_plot()
    generate_student_histogram()

    print("✅ Plots saved in static/results/ folder.")
