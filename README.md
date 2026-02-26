# 📊 Telecom Customer Churn Analysis & Prediction

End-to-End Data Analytics & Machine Learning Project

## Project Overview

Customer churn is one of the most critical revenue risks in the telecom industry.

This project delivers a complete end-to-end solution:

* 📊 Business Intelligence Dashboard (Power BI)
* 🧹 Data Cleaning & Feature Engineering (Python)
* 🤖 Machine Learning Model for Churn Prediction
* 📈 Business Insight & Revenue Impact Analysis

Dataset size: **7,043 customers**

# 📈 Executive Summary

| Metric          | Value       |
| --------------- | ----------- |
| Total Customers | 7,043       |
| Total Revenue   | $21.37M     |
| Total Refund    | $13.82K     |
| Churn Rate      | 26.54%      |
| Average Charges | 2.28K       |
| Total Tenure    | 228K Months |

# 📊 Power BI Dashboard

The interactive dashboard provides executive-level visibility into:

* Revenue by Contract Type
* Churn Category Breakdown
* Customer Distribution by Gender
* Revenue by Age Group
* Average Monthly Charges by Customer Status
* Revenue & Churn Rate by City
* Service-based filtering (Streaming, Internet Type, Premium Support)

# 🔎 Key Insights

* Month-to-month contracts show highest churn.
* Competitor switching is the primary churn driver.
* Higher monthly charges correlate with higher churn.
* Two-year contracts generate highest revenue ($9.04M).
* Churn rate: **26.54%**

# 🧹 Data Preprocessing Pipeline

# 1. Data Cleaning

* Removed inconsistencies
* Converted categorical variables
* Encoded binary features
* Handled missing values

# 2. Feature Engineering

* Created numerical encodings
* Standardized tenure & charges
* Built churn target variable

#3. Train-Test Split

* 80% Training
* 20% Testing
* Stratified sampling
* Random state = 42

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Test set size: **1,407 customers**


# 🤖 Machine Learning Models

# 1. Logistic Regression

* Solver: lbfgs
* Penalty: L2
* Scaled features using StandardScaler

# 📊 Performance

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 80.38% |
| ROC-AUC           | 0.8357 |
| Precision (Churn) | 65%    |
| Recall (Churn)    | 57%    |
| F1-Score (Churn)  | 61%    |

#Interpretation

* Strong overall accuracy
* Good probability discrimination (AUC > 0.83)
* Best balance between precision & recall



# 2. Random Forest Classifier

* n_estimators = 300
* class_weight = balanced
* random_state = 42

# 📊 Performance

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 78.75% |
| ROC-AUC           | 0.8199 |
| Precision (Churn) | 63%    |
| Recall (Churn)    | 49%    |
| F1-Score (Churn)  | 55%    |

# Interpretation

* Strong majority-class detection
* Slightly lower churn recall
* Slightly lower AUC than Logistic Regression


# 3. Hyperparameter Tuning (GridSearchCV)

* Cross-validation: 5-fold
* Scoring metric: ROC-AUC
* Tuned C parameter (0.01 – 10)
* Best AUC: 0.8352

Final optimized model remained Logistic Regression (~80% accuracy).



# 4. ROC Curve Comparison

Logistic Regression achieved the highest AUC: **0.8357**

This indicates strong class separability between churn and non-churn customers.


# 🏆 Final Model Selection : Logistic Regression**

### Why?

* Highest AUC
* Most balanced performance
* More interpretable
* Easier to deploy
* Stable across folds


# Model Deployment Preparation

Saved artifacts:

* logistic_model.pkl
* random_forest_model.pkl
* scaler.pkl
* cleaned_dataset.csv

Using:

```python
joblib.dump()
```

This makes the solution production-ready.


# 📊 Business Impact Analysis

Model predicts churn with:

* ~80% accuracy
* 57% recall for churn customers
* Strong probability discrimination

### Business Value

If deployed:

* Identify high-risk customers early
* Launch retention campaigns
* Offer contract upgrades
* Reduce churn by even 5%

A 5% churn reduction could protect **$1M+ annually** in revenue.


# 🛠️ Tech Stack

| Category      | Tools         |
| ------------- | ------------- |
| Programming   | Python        |
| Data Analysis | Pandas, NumPy |
| Visualization | Power BI      |
| ML Models     | Scikit-learn  |
| Tuning        | GridSearchCV  |
| Model Saving  | Joblib        |
| Notebook      | Jupyter       |


# 🧠 Skills Demonstrated

* Business Intelligence Development
* Data Cleaning & Preprocessing
* Feature Engineering
* Supervised Learning
* Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
* Hyperparameter Tuning
* Class Imbalance Handling
* Data Visualization
* Business Interpretation of ML Results
* End-to-End Pipeline Development

# 🔮 Future Improvements

* XGBoost / LightGBM implementation
* SMOTE for class imbalance
* Threshold optimization
* Cost-sensitive modeling
* Model deployment via Flask / FastAPI
* Power BI Service publishing
* Automated refresh pipeline

---

# 📂 Project Structure

```
├── Telcom_Customer_churn.ipynb
├── Report_BI.pdf
├── dataset.csv
├── logistic_model.pkl
├── random_forest_model.pkl
├── scaler.pkl
└── README.md
```

---

# 🎯 Why This Project Stands Out

This project demonstrates:

* Strong understanding of business problems
* Ability to translate data into revenue insights
* Full ML lifecycle implementation
* Clean model comparison methodology
* Executive-level dashboard presentation
* Production-ready mindset



# 👨‍💻 About Me

Aspiring Data Analyst / Data Scientist
Focused on transforming data into actionable business decisions.

Passionate about:

* Analytical problem solving
* Business-driven modeling
* Clear data storytelling
* Building deployable solutions
