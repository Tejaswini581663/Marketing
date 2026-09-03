import pandas as pd
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Load Cleaned Campaign Data & Aggregate Key Metrics
df = pd.read_csv('data/cleaned_marketing_ab.csv')

summary_stats = df.groupby('test group').agg(
    total_users=('user id', 'count'),
    total_conversions=('converted', 'sum'),
    avg_ads_seen=('total ads', 'mean')
).reset_index()

summary_stats['conversion_rate_%'] = (summary_stats['total_conversions'] / summary_stats['total_users']) * 100
metrics_str = summary_stats.to_string(index=False)

# 2. Define Prompt Template
prompt_template = """
You are an expert Chief Marketing Officer (CMO) AI assistant. 
Analyze the following A/B test results from our recent ad campaign:

{campaign_metrics}

Provide a concise executive summary:
1. Key findings comparing variant conversion rates.
2. Behavioral patterns regarding total ad exposure.
3. Actionable strategic recommendation on whether to roll out the 'ad' or 'psa' variant broadly.
"""

prompt = PromptTemplate(input_variables=["campaign_metrics"], template=prompt_template)

# 3. Initialize Local Ollama Model (No OpenAI Required)
llm = ChatOllama(model="llama3.2", temperature=0.3)
chain = prompt | llm

# 4. Generate Insights
print("--- Generating Local AI Executive Summary ---")
response = chain.invoke({"campaign_metrics": metrics_str})
print(response.content)

# 5. Save Summary locally for RAG indexing
os.makedirs('data', exist_ok=True)
with open('data/campaign_summary.txt', 'w', encoding='utf-8') as f:
    f.write(response.content)

print("\nExecutive summary saved to data/campaign_summary.txt")