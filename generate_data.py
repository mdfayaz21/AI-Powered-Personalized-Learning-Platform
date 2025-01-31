import pandas as pd
import numpy as np

# Define data
students = 1000
learning_styles = ['visual', 'auditory', 'kinesthetic']
subjects = ['math', 'science', 'history']
recommendations = ['advanced_math', 'basic_science', 'history_essay']

# Generate synthetic data
data = {
    'student_id': range(1, students + 1),
    'learning_style': np.random.choice(learning_styles, students),
    'subject': np.random.choice(subjects, students),
    'time_spent': np.random.randint(10, 60, students),
    'score': np.random.randint(50, 100, students),
    'recommendation': np.random.choice(recommendations, students)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('learning_data.csv', index=False)
print("Synthetic dataset generated and saved as 'learning_data.csv'!")