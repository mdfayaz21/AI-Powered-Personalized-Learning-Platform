# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
    return df

# Preprocess Data
def preprocess_data(df):
    # Encode categorical features
    le_style = LabelEncoder()
    le_subject = LabelEncoder()
    le_recommendation = LabelEncoder()

    df['learning_style'] = le_style.fit_transform(df['learning_style'])
    df['subject'] = le_subject.fit_transform(df['subject'])
    df['recommendation'] = le_recommendation.fit_transform(df['recommendation'])

    # Features and Target
    X = df.drop(['student_id', 'recommendation'], axis=1)
    y = df['recommendation']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le_style, le_subject, le_recommendation

# Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model Trained Successfully!")
    return model

# Flask App for Recommendations
app = Flask(__name__)
CORS(app)

# Load Model and Data
df = load_data("learning_data.csv")
X_train, X_test, y_train, y_test, le_style, le_subject, le_recommendation = preprocess_data(df)
model = train_model(X_train, y_train)

# Root Route
@app.route('/')
def home():
    return "Welcome to the AI-Powered Personalized Learning Platform!"

# Recommendation API
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    learning_style = data['learning_style']
    subject = data['subject']
    time_spent = data['time_spent']
    score = data['score']

    # Encode input data
    learning_style_encoded = le_style.transform([learning_style])[0]
    subject_encoded = le_subject.transform([subject])[0]

    # Predict recommendation
    input_data = np.array([[learning_style_encoded, subject_encoded, time_spent, score]])
    recommendation_encoded = model.predict(input_data)[0]
    recommendation = le_recommendation.inverse_transform([recommendation_encoded])[0]

    return jsonify({"recommendation": recommendation})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)