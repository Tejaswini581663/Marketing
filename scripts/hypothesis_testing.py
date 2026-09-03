import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

# Load cleaned data
df = pd.read_csv('data/cleaned_marketing_ab.csv')

# 1. Chi-Square Test of Independence for Conversion Rates
contingency_table = pd.crosstab(df['test group'], df['converted'])
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print("--- A/B Test Statistical Significance ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p_val:.4e}")

if p_val < 0.05:
    print("Result: Statistically Significant (Reject Null Hypothesis)")
else:
    print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

# 2. Independent Two-Sample T-Test for Engagement (total_ads seen)
ad_group = df[df['test group'] == 'ad']['total ads']
psa_group = df[df['test group'] == 'psa']['total ads']

t_stat, t_pval = ttest_ind(ad_group, psa_group, equal_var=False)
print(f"\nT-Test Statistic (Ad Exposure): {t_stat:.4f}, P-Value: {t_pval:.4e}")