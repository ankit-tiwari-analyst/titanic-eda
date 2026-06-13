# Start Week 2 Lesson 4: Full EDA notebook

# Phase 1 — Load & overview

# ==================================================
# TITANIC EDA - Full Analysis
# Author: [Ankit Tiwari]   Date: [05/06/2026]
# ==================================================


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
df = pd.read_csv('titanic.csv')

# --- 1. OVERVIEW ---
print('='*50)
print('DATASET OVERVIEW')
print('='*50)
print(f'rows:   {df.shape[0]}')
print(f'columns:   {df.shape[1]}')
print(f'columns:   {df.columns.tolist()}')
print('\nMissing values:')
print((df.isnull().sum()[df.isnull().sum() > 0]))
print('\nData Types:')
print(df.dtypes)
print('\nBasic stats:')
print(df.describe().round(2))


# Phase 2 — Clean the data

# --- 2. DATA CLEANING ---
print('\n' + '='*50)
print('DATA CLEANING')
print('='*50)

# Fill missing Age with median
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

# Drop Cabin — too many missing (77%)
df.drop(columns=['Cabin'], inplace=True)

# Drop 2 rows where Embarked is missing
df.dropna(subset=['Embarked'], inplace=True)

# Cap Fare outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
upper_fare = Q3 + 1.5 * (Q3 - Q1)
df['Fare'] = df['Fare'].clip(upper=upper_fare)

print(f'Age: filled {df['Age'].isnull().sum()} Missing with median ({age_median})')
print(f'Cabin: dropped (77% missing)')
print(f'Fare: capped at £{upper_fare:.2f}')
print(f'Clean Dataset: {df.shape[0]} rows, {df.shape[1]} columns')


# Phase 3 — Univariate analysis

# --- 3. UNIVARIATE ANALYSIS ---
fig, axes = plt.subplots(2, 2, figsize=(14,10))
fig.suptitle('Univariate Analysis — Titanic', fontsize=16, fontweight='bold')

# Age distribution
sns.histplot(df['Age'], bins=30, ax=axes[0,0], color='#3498db')
axes[0,0].set_title('Age Distribution')
axes[0,0].axvline(df['Age'].median(), color='red', linestyle='--', label='Median')
axes[0,0].legend()

# Fare distribution
sns.histplot(df['Fare'], bins=30, ax=axes[0,1], color='#9b59b6')
axes[0,1].set_title('Fare Distribution (Capped)')
axes[0,1].axvline(df['Fare'].median(), color='red', linestyle='--', label='Median')
axes[0,1].legend()

# Survival count
sns.countplot(x='Survived', data=df, ax=axes[1,0],
              palette='Set2', hue='Survived', legend=False)
axes[1,0].set_title('Survived vs Died')
axes[1,0].set_xticklabels(['Died', 'Survived'])

# Passenger class count
sns.countplot(x='Pclass', data=df, ax=axes[1,1],
              palette='viridis', hue='Pclass', legend=False)
axes[1,1].set_title('Passengers by Class')
axes[1,1].set_xticklabels(['1st', '2nd', '3rd'])

plt.tight_layout()
plt.savefig('eda_univariate.png', dpi=150, bbox_inches='tight')
plt.show()


# Phase 4 — Bivariate analysis

# --- 4. BIVARIATE ANALYSIS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bivariate Analysis — Titanic', fontsize=16, fontweight='bold')

# Survival by class and sex
sns.barplot(data=df, x='Pclass', y='Survived',
            hue='Sex', palette='Set1', ax=axes[0,0])
axes[0,0].set_title('Survival Rate by Class & Sex')
axes[0,0].set_xticklabels(['1st', '2nd', '3rd'])

# Age by class boxplot
sns.boxplot(data=df, x='Pclass', y='Age',
            palette='coolwarm', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Class')
axes[0,1].set_xticklabels(['1st', '2nd', '3rd'])

# Fare by survival
sns.boxplot(data=df, x='Survived', y='Fare',
            palette='Set2', ax=axes[1,0])
axes[1,0].set_title('Fare Distribution by Survival')
axes[1,0].set_xticklabels(['Died', 'Survived'])

# Correlation heatmap
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_bivariate.png', dpi=150, bbox_inches='tight')
plt.show()


# Phase 5 — Written findings summary

# --- 5. KEY FINDINGS ---
surv_rate    = df['Survived'].mean() * 100
female_surv  = df[df['Sex']=='female']['Survived'].mean() * 100
male_surv    = df[df['Sex']=='male']['Survived'].mean() * 100
class1_surv  = df[df['Pclass']==1]['Survived'].mean() * 100
class3_surv  = df[df['Pclass']==3]['Survived'].mean() * 100
child_surv   = df[df['Age']<18]['Survived'].mean() * 100

print("="*50)
print("  KEY FINDINGS — TITANIC EDA")
print("="*50)
print(f"  1. Overall survival rate:      {surv_rate:.1f}%")
print(f"  2. Female survival rate:       {female_surv:.1f}%")
print(f"  3. Male survival rate:         {male_surv:.1f}%")
print(f"  4. 1st class survival rate:    {class1_surv:.1f}%")
print(f"  5. 3rd class survival rate:    {class3_surv:.1f}%")
print(f"  6. Children survival rate:     {child_surv:.1f}%")
print("="*50)
print("""
  CONCLUSIONS:
  - Gender was the strongest survival predictor.
    Women survived at 4x the rate of men.
  - Class amplified the gender effect — 1st class
    females survived at 97%, 3rd class males at 13%.
  - Children had better odds than adult males but
    worse than adult females — age mattered less
    than gender and class combined.
  - Fare correlates with class, not survival directly.
    Wealth bought a better cabin, not a lifeboat seat.
""")
print("="*50)


# Lesson 8 Assignments

# Phase 1: Load titanic.csv — print full overview block (shape, missing, dtypes, describe)

# ==================================================
# TITANIC EDA - Full Analysis
# Author: [Ankit Tiwari]   Date: [13/06/2026]
# ==================================================


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
df = pd.read_csv('titanic.csv')

# --- 1. OVERVIEW ---
print('='*50)
print('DATASET OVERVIEW')
print('='*50)
print(f'rows:   {df.shape[0]}')
print(f'columns:   {df.shape[1]}')
print(f'columns:   {df.columns.tolist()}')
print('\nMissing values:')
print((df.isnull().sum()[df.isnull().sum() > 0]))
print('\nData Types:')
print(df.dtypes)
print('\nBasic stats:')
print(df.describe().round(2))

# Phase 2: Clean — fill Age with median, drop Cabin, drop Embarked nulls, cap Fare outliers. Print what you did.

# --- 2. DATA CLEANING ---
print('\n' + '='*50)
print('DATA CLEANING')
print('='*50)

# Fill missing Age with median
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

# Drop Cabin — too many missing (77%)
df.drop(columns=['Cabin'], inplace=True)

# Drop 2 rows where Embarked is missing
df.dropna(subset=['Embarked'], inplace=True)

# Cap Fare outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
upper_fare = Q3 + 1.5 * (Q3 - Q1)
df['Fare'] = df['Fare'].clip(upper=upper_fare)

print(f'Age: filled {df['Age'].isnull().sum()} Missing with median ({age_median})')
print(f'Cabin: dropped (77% missing)')
print(f'Fare: capped at £{upper_fare:.2f}')
print(f'Clean Dataset: {df.shape[0]} rows, {df.shape[1]} columns')

# Phase 3: Univariate — 2×2 grid of charts (Age hist, Fare hist, Survived countplot, Pclass countplot). Save as PNG.

# --- 3. UNIVARIATE ANALYSIS ---
fig, axes = plt.subplots(2, 2, figsize=(14,10))
fig.suptitle('Univariate Analysis — Titanic', fontsize=16, fontweight='bold')

# Age distribution
sns.histplot(df['Age'], bins=30, ax=axes[0,0], color='#3498db')
axes[0,0].set_title('Age Distribution')
axes[0,0].axvline(df['Age'].median(), color='red', linestyle='--', label='Median')
axes[0,0].legend()

# Fare distribution
sns.histplot(df['Fare'], bins=30, ax=axes[0,1], color='#9b59b6')
axes[0,1].set_title('Fare Distribution (Capped)')
axes[0,1].axvline(df['Fare'].median(), color='red', linestyle='--', label='Median')
axes[0,1].legend()

# Survival count
sns.countplot(x='Survived', data=df, ax=axes[1,0],
              palette='Set2', hue='Survived', legend=False)
axes[1,0].set_title('Survived vs Died')
axes[1,0].set_xticklabels(['Died', 'Survived'])

# Passenger class count
sns.countplot(x='Pclass', data=df, ax=axes[1,1],
              palette='viridis', hue='Pclass', legend=False)
axes[1,1].set_title('Passengers by Class')
axes[1,1].set_xticklabels(['1st', '2nd', '3rd'])

plt.tight_layout()
plt.savefig('eda_univariate.png', dpi=150, bbox_inches='tight')
plt.show()

# Phase 4: Bivariate — 2×2 grid (survival by class+sex barplot, age boxplot by class, fare by survival boxplot, heatmap). Save as PNG.

# --- 4. BIVARIATE ANALYSIS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bivariate Analysis — Titanic', fontsize=16, fontweight='bold')

# Survival by class and sex
sns.barplot(data=df, x='Pclass', y='Survived',
            hue='Sex', palette='Set1', ax=axes[0,0])
axes[0,0].set_title('Survival Rate by Class & Sex')
axes[0,0].set_xticklabels(['1st', '2nd', '3rd'])

# Age by class boxplot
sns.boxplot(data=df, x='Pclass', y='Age',
            palette='coolwarm', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Class')
axes[0,1].set_xticklabels(['1st', '2nd', '3rd'])

# Fare by survival
sns.boxplot(data=df, x='Survived', y='Fare',
            palette='Set2', ax=axes[1,0])
axes[1,0].set_title('Fare Distribution by Survival')
axes[1,0].set_xticklabels(['Died', 'Survived'])

# Correlation heatmap
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_bivariate.png', dpi=150, bbox_inches='tight')
plt.show()

# Phase 5: Written findings — compute and print at least 6 survival stats with a conclusions paragraph in your own words.

# --- 5. KEY FINDINGS ---
surv_rate    = df['Survived'].mean() * 100
female_surv  = df[df['Sex']=='female']['Survived'].mean() * 100
male_surv    = df[df['Sex']=='male']['Survived'].mean() * 100
class1_surv  = df[df['Pclass']==1]['Survived'].mean() * 100
class3_surv  = df[df['Pclass']==3]['Survived'].mean() * 100
child_surv   = df[df['Age']<18]['Survived'].mean() * 100

print("="*50)
print("  KEY FINDINGS — TITANIC EDA")
print("="*50)
print(f"  1. Overall survival rate:      {surv_rate:.1f}%")
print(f"  2. Female survival rate:       {female_surv:.1f}%")
print(f"  3. Male survival rate:         {male_surv:.1f}%")
print(f"  4. 1st class survival rate:    {class1_surv:.1f}%")
print(f"  5. 3rd class survival rate:    {class3_surv:.1f}%")
print(f"  6. Children survival rate:     {child_surv:.1f}%")
print("="*50)
print("""
  CONCLUSIONS:
  - Gender was the strongest survival predictor.
    Women survived at 4x the rate of men.
  - Class amplified the gender effect — 1st class
    females survived at 97%, 3rd class males at 13%.
  - Children had better odds than adult males but
    worse than adult females — age mattered less
    than gender and class combined.
  - Fare correlates with class, not survival directly.
    Wealth bought a better cabin, not a lifeboat seat.
""")
print("="*50)

# Add your name and date as a comment at the top — this is your portfolio piece.

# it is done at the top of the EDA 

'''CHALLENGE: Add a 5th chart to either grid — a sns.scatterplot of Age vs Fare colored by Survived (use hue='Survived'). 
Write one sentence explaining what pattern you see.'''

fig, ax = plt.subplots(figsize=(10,8))

sns.scatterplot(
    data=df,
    x = 'Age',
    y = 'Fare',
    hue = 'Survived',
    palette = {0: '#e74c3c', 1: '#2ecc71'},
    alpha = 0.6,
    ax = ax
)

ax.set_title('Age vs Fare — Colored by Survival', fontsize=14, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Fare (£)')
ax.legend(title='Survived', labels=['Died', 'Survived'])

plt.tight_layout()
plt.savefig('eda_scatter_survival.png', dpi=150, bbox_inches='tight')
plt.show()

'''OBSERVATION: Survivors (green) are concentrated at higher fare levels across all ages, 
while non-survivors (red) dominate the low-fare region — confirming that fare (and # therefore class) was a stronger survival predictor than age alone.'''


