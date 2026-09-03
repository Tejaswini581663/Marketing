import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

os.makedirs('models', exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('data/cleaned_marketing_ab.csv')

# 2. Features & Target
categorical_features = ['test group', 'most ads day']
numerical_features = ['total ads', 'most ads hour']

feature_cols = categorical_features + numerical_features
X = df[feature_cols]
y = df['converted']

# 3. Train-Test Split with Class Check
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # Fallback split without stratification if class count is low
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# 5. Define ML Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# 6. Fit Model
model_pipeline.fit(X_train, y_train)

# 7. Model Evaluation Metrics
y_pred = model_pipeline.predict(X_test)
print("--- Model Performance Metrics ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Save Model
joblib.dump(model_pipeline, 'models/conversion_random_forest.pkl')
print("Trained model pipeline saved to models/conversion_random_forest.pkl")