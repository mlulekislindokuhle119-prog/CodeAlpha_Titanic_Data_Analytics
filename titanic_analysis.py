# File Name: titanic_analysis.py
# Description: Complete script for CodeAlpha Data Analytics Internship - Task 2 (EDA) and Task 3 (Data Visualization)[cite: 1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ==========================================
# Load the Titanic dataset from the public repository specified for the analysis
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Set visual styling for all generated charts using seaborn themes
sns.set_theme(style="whitegrid")

print("--- DATASET STRUCTURAL INFORMATION ---")
print(df.info())

print("\n--- IDENTIFYING MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DESCRIPTIVE STATISTICS: NUMERICAL VARIABLES ---")
print(df.describe())

print("\n--- DESCRIPTIVE STATISTICS: CATEGORICAL VARIABLES ---")
print(df.describe(include=['O']))

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
# Address missing values in the 'Age' column by imputing with the median age to mitigate outlier skewness
df['Age'] = df['Age'].fillna(df['Age'].median())

# Address missing values in the 'Embarked' column by filling with the statistical mode (most frequent port)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ==========================================
# 3. QUANTITATIVE STATISTICAL BREAKDOWNS
# ==========================================
print("\n--- TARGETED STATISTICAL METRICS FOR REPORTING ---")
overall_survival_rate = df['Survived'].mean() * 100
print(f"Overall Passenger Survival Rate: {overall_survival_rate:.2f}%")

gender_survival = (df.groupby('Sex')['Survived'].mean() * 100).round(2)
print("\nSurvival Rate Breakdown by Gender (%):")
print(gender_survival)

class_survival = (df.groupby('Pclass')['Survived'].mean() * 100).round(2)
print("\nSurvival Rate Breakdown by Passenger Class (%):")
print(class_survival)

mean_fare_by_class = df.groupby('Pclass')['Fare'].mean().round(2)
print("\nAverage Ticket Fare by Passenger Class:")
print(mean_fare_by_class)

# ==========================================
# 4. DATA VISUALIZATION DASHBOARD (TASK 3)
# ==========================================
# Constructing a multi-panel 2x2 subplot layout to encapsulate multidimensional trends per CodeAlpha guidelines[cite: 1]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Visualization 1: Overall Survival Count Distribution
sns.countplot(x='Survived', data=df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Overall Survival Count (0 = Died, 1 = Survived)')
axes[0, 0].set_xticklabels(['Died', 'Survived'])
axes[0, 0].set_xlabel('Survival Status')
axes[0, 0].set_ylabel('Passenger Count')

# Visualization 2: Disparate Survival Rates Categorized by Gender
sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0, 1], palette='Set1', errorbar=None)
axes[0, 1].set_title('Survival Rate by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Survival Probability')

# Visualization 3: Intersection of Passenger Class, Gender, and Survival Likelihood
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=axes[1, 0], palette='Blues_d', errorbar=None)
axes[1, 0].set_title('Survival Rate by Passenger Class & Gender')
axes[1, 0].set_xlabel('Passenger Class (Pclass)')
axes[1, 0].set_ylabel('Survival Probability')

# Visualization 4: Comparative Age Distribution Profiling Survivors vs Non-Survivors
sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[1, 1], palette='husl', bins=30)
axes[1, 1].set_title('Age Distribution of Survivors vs Non-Survivors')
axes[1, 1].set_xlabel('Age (Years)')
axes[1, 1].set_ylabel('Density / Count')

# Finalize layout adjustments and render output display
plt.tight_layout()
plt.show()