import pandas as pd
import sqlite3
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Connect to SQLite database
conn = sqlite3.connect('Data/marketing.db')

# Read full table and sanitize column names (replaces spaces with underscores)
df = pd.read_sql_query("SELECT * FROM marketing_campaign", conn)
conn.close()

df.columns = [c.replace(' ', '_') for c in df.columns]

# 2. Map test group values ('ad' -> 1, 'psa' -> 0)
df['test_group'] = df['test_group'].map({'ad': 1, 'psa': 0})

# 3. Define features and target variable
X = df[['test_group', 'total_ads', 'most_ads_hour']]
y = df['converted']

# 4. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save serialized model using joblib
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/conversion_random_forest.joblib', protocol=4)
print("Model successfully trained and saved to models/conversion_random_forest.joblib!")