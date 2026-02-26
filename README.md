Telecom Customer Churn Prediction
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-yellow?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/SQL%20Server-ETL-CC2927?style=for-the-badge&logo=microsoftsqlserver&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>
Predict which telecom customers are about to leave — a full end-to-end pipeline from SQL ETL to Python ML to Power BI dashboard that turns churn risk into actionable retention strategy.

Problem Statement
In telecom, acquiring a new customer costs 5 to 10 times more than retaining one. With a 26.54% churn rate across 7,043 customers, identifying at-risk customers before they leave is a critical business need. This project delivers a complete predictive solution from raw data ingestion to stakeholder-ready dashboards.

What This Project Does

SQL Server ETL pipeline — stages, transforms, and curates raw CSV data for analysis
Exploratory Data Analysis — uncovers churn drivers across contract type, tenure, and payment method
ML models trained and compared — Logistic Regression vs Random Forest with hyperparameter tuning
Interactive Power BI dashboard — KPI cards, geographic churn map, and revenue analysis
Saved model artifacts — .pkl files ready for deployment or inference on new customers


Project Structure
Telecom-customer-churn-prediction/
├── Telcom_Customer_churn.ipynb       # Full ML pipeline notebook
├── RandomForestClassifier.pkl        # Saved Random Forest model
├── LogisticRegression.pkl            # Saved Logistic Regression model (FINAL)
├── scaler.pkl                        # Fitted StandardScaler
├── churn_prediction_data.csv         # Cleaned and processed output dataset
├── sql/
│   └── etl_pipeline.sql              # SQL Server ETL pipeline
├── Report_BI.pbix                    # Interactive Power BI dashboard
└── README.md
Source dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv — IBM Telco, 7,043 rows, 21 features

Stage 1 — SQL Server ETL Pipeline
Raw data was processed through a structured ETL pipeline before any ML work.
CSV File  →  Staging Table  →  Transformation  →  Final Table  →  Analytics Views
Steps performed in SQL Server (SSMS):

Created staging and final schema tables
Loaded raw CSV via flat file import
Applied data transformations and quality checks
Built analytical views for churn rate by contract type
Output: clean, analysis-ready dataset passed into Python


Stage 2 — Exploratory Data Analysis
Dataset: 7,043 customers. Churn rate: 26.54%. Moderate class imbalance — ROC-AUC prioritized over accuracy.
Key FindingInsightContract typeMonth-to-Month customers churn far more than 1-year or 2-year holdersTenureChurned customers have significantly lower average tenurePayment methodElectronic check users churn at the highest rateMonthly chargesChurned avg $75/month vs Stayed avg $63/monthSenior citizensAround 16% of base — distinct behavioral segment

Stage 3 — Machine Learning Pipeline
Preprocessing Steps
StepMethodTarget encodingChurn: Yes = 1, No = 0Dropped columncustomerID (not predictive)TotalCharges fixConverted to numeric, 11 NaN rows droppedCategorical encodingpd.get_dummies() with drop_first=TrueFeature scalingStandardScaler — fit on train, transform on testTrain/test split80/20, random_state=42, stratify=y
Models Trained
python# Logistic Regression
LogisticRegression(max_iter=1000)

# Random Forest
RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
Hyperparameter Tuning
GridSearchCV on Logistic Regression — 5-fold CV, scoring = roc_auc:
pythonparam_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10],
    'penalty': ['l2']
}
Model Comparison
ModelROC-AUCNotesLogistic Regression (baseline)0.835Strong interpretable baselineRandom Forest0.819Higher complexity, lower AUCLogistic Regression (tuned)0.835Final model selected
Final model: Tuned Logistic Regression. Decision threshold adjusted to 0.4 to improve churn recall to 57%.

Saved Model Artifacts
Three files saved with joblib:
LogisticRegression.pkl       — FINAL prediction model
RandomForestClassifier.pkl   — feature importance reference
scaler.pkl                   — StandardScaler (required for Logistic Regression)
Load and predict on new data:
pythonimport joblib

model  = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")

X_scaled   = scaler.transform(X_new)
churn_prob = model.predict_proba(X_scaled)[:, 1]
churn_flag = (churn_prob > 0.4).astype(int)

Stage 4 — Power BI Dashboard
File: Report_BI.pbix — Open with Power BI Desktop (free)
KPI Cards
MetricValueTotal Customers7,043Total Revenue$21.37MTotal Refund$13.82KAverage Charges$2.28KTotal Tenure (months)228KChurn Rate26.54%
Dashboard Visuals
VisualWhat It ShowsRevenue by Contract (Donut)Month-to-Month $6.16M, One Year $6.17M, Two Year $9.04MChurn Category (Bar)Competitor 841, Dissatisfaction 321, Attitude 314, Price 211, Other 182Customers by Gender (Pie)Male 50.48%, Female 49.52%Customers and Revenue by AgeAbove-60 is the largest customer and revenue segmentAvg Monthly Charges by StatusChurned $75, Stayed $63, Joined $44Geographic MapRevenue and churn rate across California cities
Interactive Filters: Streaming TV, Streaming Music, Streaming Movie, Unlimited Data, Internet Service, Internet Type (Cable / DSL / Fiber Optic), Premium Support

Tech Stack
LayerToolsETLSQL Server, SSMS, CSV Flat File ImportLanguagePython — Google Colab / JupyterData WranglingPandas, NumPyVisualizationMatplotlib, SeabornMachine LearningScikit-learn — LogisticRegression, RandomForestClassifier, GridSearchCVModel PersistenceJoblibBI DashboardPower BI Desktop

Getting Started
1. Clone the repository
bashgit clone https://github.com/shreevarsha866/Telecom-customer-churn-prediction.git
cd Telecom-customer-churn-prediction
2. Install dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn joblib
3. Run the notebook
Open Telcom_Customer_churn.ipynb in Jupyter or Google Colab. Place WA_Fn-UseC_-Telco-Customer-Churn.csv at /content/ for Colab or update the path in Cell 1.
4. Predict on new customers
pythonimport joblib

model  = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")

X_scaled   = scaler.transform(X_new)
churn_prob = model.predict_proba(X_scaled)[:, 1]
at_risk    = (churn_prob > 0.4).astype(int)
5. Open the Power BI dashboard
Open Report_BI.pbix in Power BI Desktop and explore with the interactive filters.

Final Results
MetricValueFinal ModelLogistic Regression (GridSearchCV tuned)Accuracy80%ROC-AUC Score0.835Churn Recall (threshold = 0.4)57%Dataset Size7,043 customersChurn Rate26.54%

Business Recommendations

Competitor churn is the top reason (841 cases) — launch targeted competitive pricing campaigns immediately
Month-to-month customers are highest risk — incentivize upgrades to annual contracts early in tenure
High-charge new customers are most vulnerable — churned customers pay $12/month more on average
Above-60 age group generates the most revenue — prioritize premium support offerings for this segment
Use a threshold of 0.4 instead of 0.5 when scoring customers — improves recall by flagging more true churners


About
Shreevarsha — Data Science enthusiast building end-to-end solutions from raw data to business impact.
Show Image
Show Image

License
This project is licensed under the MIT License.
