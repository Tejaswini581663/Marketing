import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
file_path = 'data/raw_marketing_ab.csv'

# Generate 100 rows of synthetic sample data for ML split
np.random.seed(42)
n_samples = 100

sample_df = pd.DataFrame({
    'user id': range(1, n_samples + 1),
    'test group': np.random.choice(['ad', 'psa'], size=n_samples),
    'converted': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
    'total ads': np.random.randint(1, 50, size=n_samples),
    'most ads day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], size=n_samples),
    'most ads hour': np.random.randint(0, 24, size=n_samples)
})

sample_df.to_csv('data/cleaned_marketing_ab.csv', index=False)
print("Updated cleaned_marketing_ab.csv with 100 sample rows!")