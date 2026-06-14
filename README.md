# Titanic EDA — Exploratory Data Analysis

## Overview
An end-to-end exploratory data analysis of the Titanic
passenger dataset (891 rows, 12 columns) using Python.

## Key Questions Answered
- What factors most influenced survival?
- Did gender, age, or class matter most?
- How are fares distributed across passenger classes?

## Key Findings
- **38.4%** of passengers survived overall
- **74.2%** of women survived vs **18.9%** of men
- **1st class** survival: 63% | **3rd class**: 24%
- Fare and class are strongly negatively correlated (-0.55)
- Wealth bought a better cabin, not a lifeboat seat

## Tools Used
- Python 3.x
- Pandas — data manipulation
- Matplotlib & Seaborn — visualisation
- NumPy — numerical operations

## Charts Produced
| Chart | Description |
|-------|-------------|
| eda_univariate.png | Age, Fare, Survival, Class distributions |
| eda_bivariate.png  | Survival by class/sex, correlations |
| eda_scatter_survival.png | Age vs Fare coloured by survival |

## How to Run
```bash
pip install pandas matplotlib seaborn numpy
python lesson8.py
```

## Author
Ankit Tiwari — Data Analyst in training
[LinkedIn](https://www.linkedin.com/in/ankit-tiwari-b9b7473b1) | New Delhi, India
