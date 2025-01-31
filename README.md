# AI-Powered-Personalized-Learning-Platform
Develop an AI-based platform that personalizes learning content for students based on their learning style, pace, and performance.
Tech Stack: Python, TensorFlow/PyTorch, Flask/Django, React.js, MongoDB.

Features:

Adaptive learning algorithms.

Real-time progress tracking.

Recommendation system for learning materials.

Explanation of the Code
Data Preprocessing:

Categorical features (learning_style, subject, recommendation) are encoded using LabelEncoder.

Model Training:

A Random Forest Classifier is used to predict the recommendation based on the student's learning style, subject, time spent, and score.

Flask API:

The API takes input data (learning style, subject, time spent, score) and returns a personalized recommendation.
