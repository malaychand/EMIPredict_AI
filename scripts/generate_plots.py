# scripts/generate_plots.py

"""
GENERATE PLOTS SCRIPT
--------------------
Tasks:
1. Load cleaned EMI dataset
2. Generate visual EDA plots:
   - Target distribution
   - Numeric feature histograms
   - Correlation heatmap
   - Derived ratios distributions
   - Feature-target correlations
   - Pairplots of key features
   - Scatter plots of key features vs ratios
3. Save all plots to reports/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIGURATION
# -----------------------------
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = ARTIFACTS_DIR / "cleaned_EMI_dataset.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CLEAN_FILE)
sns.set(style="whitegrid")
print(f"📥 Loaded cleaned dataset: {CLEAN_FILE} with shape {df.shape}")

# -----------------------------
# 1. EMI Eligibility Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='emi_eligibility', palette='Set2')
plt.title("EMI Eligibility Class Distribution")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "emi_eligibility_distribution.png")
plt.close()
print("✅ EMI eligibility distribution saved.")

# -----------------------------
# 2. Histograms of Numeric Features
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
n_cols = 4
n_rows = math.ceil(len(num_cols) / n_cols)

plt.figure(figsize=(5*n_cols, 4*n_rows))
for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "numeric_features_histograms.png")
plt.close()
print("✅ Numeric feature histograms saved.")

# -----------------------------
# 3. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12,10))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_matrix.png")
plt.close()
print("✅ Correlation matrix saved.")

# -----------------------------
# 4. Debt-to-Income Ratio Distribution
# -----------------------------
df['debt_to_income'] = df['current_emi_amount'] / df['monthly_salary'].replace(0, np.nan)
plt.figure(figsize=(6,4))
sns.histplot(df['debt_to_income'].dropna(), kde=True)
plt.title("Debt-to-Income Ratio Distribution")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "debt_to_income_distribution.png")
plt.close()
print("✅ Debt-to-Income ratio distribution saved.")

# -----------------------------
# 5. Expense-to-Income Ratio Distribution
# -----------------------------
expense_cols = [
    'school_fees','college_fees','travel_expenses',
    'groceries_utilities','other_monthly_expenses','monthly_rent'
]
df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
df['expense_to_income'] = df['total_monthly_expenses'] / df['monthly_salary'].replace(0, np.nan)

plt.figure(figsize=(6,4))
sns.histplot(df['expense_to_income'].dropna(), kde=True)
plt.title("Expense-to-Income Ratio Distribution")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "expense_to_income_distribution.png")
plt.close()
print("✅ Expense-to-Income ratio distribution saved.")

# -----------------------------
# 6. Affordability Ratio Distribution
# -----------------------------
df['monthly_disposable'] = df['monthly_salary'] - df['total_monthly_expenses'] - df['current_emi_amount']
df['instalment_if_approved'] = df['requested_amount'] / df['requested_tenure'].replace(0, np.nan)
df['affordability_ratio'] = df['monthly_disposable'] / df['instalment_if_approved'].replace(0, np.nan)

plt.figure(figsize=(6,4))
sns.histplot(df['affordability_ratio'].dropna(), kde=True)
plt.title("Affordability Ratio Distribution")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "affordability_ratio_distribution.png")
plt.close()
print("✅ Affordability ratio distribution saved.")

# -----------------------------
# 7. Correlation with Target
# -----------------------------
le = LabelEncoder()
df['emi_eligibility_num'] = le.fit_transform(df['emi_eligibility'])

plt.figure(figsize=(12,8))
corr_target = df[num_cols + ['emi_eligibility_num']].corr()['emi_eligibility_num'].sort_values(ascending=False)
sns.barplot(x=corr_target.index, y=corr_target.values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Correlation of Numeric Features with EMI Eligibility")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "feature_target_correlation.png")
plt.close()
print("✅ Feature-target correlation saved.")

# -----------------------------
# 8. Pairplot of Key Features (sampled)
# -----------------------------
key_features = [
    'monthly_salary', 'current_emi_amount', 'credit_score', 
    'requested_amount', 'affordability_ratio', 'emi_eligibility'
]

SAMPLE_SIZE = min(10000, len(df))
df_sample = df[key_features].sample(n=SAMPLE_SIZE, random_state=42)

sns.set(style="ticks")
pairplot_fig = sns.pairplot(df_sample, hue='emi_eligibility', corner=True, diag_kind='kde', palette='Set2')
pairplot_fig.savefig(REPORTS_DIR / "pairplot_key_features.png")
plt.close()
print("✅ Pairplot of key features saved.")

# -----------------------------
# 9. Scatterplots of Key Features vs Ratios
# -----------------------------
scatter_features = ['monthly_salary', 'current_emi_amount', 'requested_amount', 'credit_score']
ratio_features = ['debt_to_income', 'expense_to_income', 'affordability_ratio']

for feat in scatter_features:
    for ratio in ratio_features:
        plt.figure(figsize=(6,4))
        sns.scatterplot(
            data=df.sample(n=SAMPLE_SIZE, random_state=42),
            x=feat,
            y=ratio,
            hue='emi_eligibility',
            alpha=0.6,
            palette='Set2'
        )
        plt.title(f"{ratio} vs {feat} by EMI Eligibility")
        plt.tight_layout()
        filename = REPORTS_DIR / f"{ratio}_vs_{feat}_scatter.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved scatter plot: {filename.name}")

print(f"\n🎯 All plots generated and saved to: {REPORTS_DIR.resolve()}")
