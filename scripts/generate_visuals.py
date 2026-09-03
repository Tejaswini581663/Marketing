import os
import matplotlib
matplotlib.use('Agg')  # Prevents GUI display issues in non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Create reports directory
os.makedirs('reports', exist_ok=True)

# 2. Load cleaned data
df = pd.read_csv('Data/cleaned_marketing_ab.csv')

# 3. Chart 1: Conversion Rate by Variant
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='test group', y='converted', ci=None, palette='viridis')
plt.title('Conversion Rate by Test Group')
plt.xlabel('Test Group')
plt.ylabel('Conversion Rate')
plt.tight_layout()
plt.savefig('reports/conversion_by_variant.png', dpi=300)
plt.close()

# 4. Chart 2: Hourly Engagement Heatmap
plt.figure(figsize=(10, 6))
pivot_table = df.pivot_table(index='most ads day', columns='most ads hour', values='converted', aggfunc='mean')
sns.heatmap(pivot_table, cmap='Blues', annot=False)
plt.title('Conversion Rate by Day and Hour')
plt.tight_layout()
plt.savefig('reports/hourly_engagement_heatmap.png', dpi=300)
plt.close()

# 5. Chart 3: Overall AB Performance Summary
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='test group', hue='converted', palette='Set2')
plt.title('Total Conversions Count by Group')
plt.tight_layout()
plt.savefig('reports/ab_test_performance_summary.png', dpi=300)
plt.close()

print("All visual reports successfully generated and overwritten in /reports!")