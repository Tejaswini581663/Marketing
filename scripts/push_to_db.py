import sqlite3
import pandas as pd
import os

# 1. Connect to SQLite database
db_path = 'Data/marketing.db'
os.makedirs('Data', exist_ok=True)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. Execute schema.sql to create/reset the table structure
with open('sql/schema.sql', 'r') as f:
    schema_script = f.read()

cursor.executescript(schema_script)
print("Database schema successfully applied from sql/schema.sql!")

# 3. Load cleaned CSV data into the database
df = pd.read_csv('Data/cleaned_marketing_ab.csv')

# Append data into the pre-defined table
df.to_sql('marketing_campaign', conn, if_exists='append', index=False)
print(f"Successfully loaded {len(df)} rows into 'marketing_campaign' table in marketing.db!")

conn.close()