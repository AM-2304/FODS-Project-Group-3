# Customer Churn Prediction using Machine Learning (IBM Telco Dataset)

## 1. Project Overview

This project aims to develop and evaluate machine learning models that predict customer churn for a telecommunications company using the publicly available IBM Telco Customer Churn dataset. The primary business objective is to identify high-risk customers before they discontinue services and to design data-driven retention strategies that maximize profit and customer lifetime value (CLV).

The project incorporates data preprocessing, exploratory data analysis (EDA), feature engineering, model development, hyperparameter tuning, profit-aware evaluation, and explainability using SHAP. The workflow demonstrates how machine learning can be used not only to predict churn but also to inform targeted interventions for customer retention.



## 2. Business Objective

The key business goal is to minimize revenue loss from customer attrition by identifying customers with a high probability of churn.

* **Primary KPI:** Expected Monthly Recurring Revenue (MRR) retained per retention campaign dollar.
* **Secondary Metrics:** Overall churn rate reduction, retention campaign ROI, precision at top K, and uplift in customer lifetime value.



## 3. Dataset Description

**Source:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size:** ~7,000 records
**Target variable:** `Churn` (Yes/No)

**Feature groups:**

* **Customer Demographics:** Gender, SeniorCitizen, Partner, Dependents
* **Account Information:** Tenure, Contract, PaymentMethod, PaperlessBilling
* **Service Subscriptions:** InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.
* **Charges:** MonthlyCharges, TotalCharges

The dataset includes a mix of numerical and categorical features, requiring encoding, scaling, and imputation as part of preprocessing.



## 4. Methodology and Workflow

### Step 1. Data Preprocessing and Feature Engineering

* Missing value imputation (median for numeric, constant for categorical)
* Outlier analysis and correction
* Encoding categorical variables using One-Hot Encoding
* Feature scaling with StandardScaler
* CLV estimation (`MonthlyCharges × remaining contract months`)
* Train-test split with stratified sampling

### Step 2. Exploratory Data Analysis (EDA)

* Visualized churn distribution across categorical and numerical attributes
* Identified strong churn drivers (Contract Type, Tenure, TechSupport, InternetService)

### Step 3. Baseline and Tuned Modeling

Models implemented:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. AdaBoost
5. XGBoost
6. LightGBM
7. CatBoost
8. TabPFN (Experimental)

Each model was trained, cross-validated, and evaluated using ROC-AUC, precision, recall, F1, and accuracy metrics.

### Step 4. Hyperparameter Optimization with Optuna

* Automated search for CatBoost and HistGradientBoosting models
* Objective function: maximize cross-validated ROC-AUC
* GPU acceleration enabled for CatBoost (T4 GPU in Colab/Kaggle environment)

### Step 5. Profit-Aware Evaluation

Implemented a custom evaluation function:
[
EV_i = P(\text{churn})*i \times (\text{CLV}*{lost,i} \times P(\text{success})) - \text{campaign cost}
]

* Simulated marketing interventions for top-K customers
* Computed expected retained revenue, ROI, and optimal targeting threshold

### Step 6. Clustering and Customer Segmentation

* PCA-based dimensionality reduction
* K-Means clustering pipeline for customer segmentation
* Cluster profiling for retention strategy mapping

### Step 7. Explainability (SHAP)

* SHAP TreeExplainer applied to the best model (CatBoost)
* Feature importance and SHAP summary plots used to interpret churn drivers
* Key churn drivers identified: Contract type, tenure, MonthlyCharges, and TechSupport availability



## 5. Tools and Technologies

* **Programming Language:** Python (3.11)
* **Libraries:**

  * Data Handling: pandas, numpy
  * Visualization: matplotlib, seaborn, shap
  * Machine Learning: scikit-learn, catboost, lightgbm, xgboost
  * Optimization: optuna
  * Persistence: joblib
* **Environment:** Google Colab / Kaggle GPU (T4)
* **Version Control:** GitHub



## 6. Key Deliverables

* Cleaned and preprocessed dataset
* Machine learning pipeline with preprocessing, modeling, and evaluation
* Hyperparameter-tuned CatBoost and HistGradientBoosting models
* Profit-aware evaluation function and ROI vs K curve
* Customer segmentation and retention strategy mapping
* SHAP-based model explainability visualizations
* Final churn prediction CSV for submission



## 7. Results Summary

* **Best model:** CatBoost (GPU)
* **Cross-validated ROC-AUC:** ~0.86 (post-tuning)
* **Top churn predictors:** Contract type, tenure, TechSupport, MonthlyCharges
* **Optimal targeting:** Top 20% customers by expected value yielded ~3.5× ROI in campaign simulation.



## 8. Folder Structure

```
├── notebooks/
│   ├── churn_modeling.ipynb
│   ├── optuna_tuning.ipynb
│   ├── profit_evaluation.ipynb
│
├── data/
│   ├── Telco_customer_churn.xlsx
│   ├── processed_telco.csv
│
├── models/
│   ├── catboost_optuna_best.joblib
│   ├── histgb_optuna_best.joblib
│
├── outputs/
│   ├── churn_predictions.csv
│   ├── shap_summary_bar.png
│   ├── roi_vs_k_plot.png
│
└── README.md
```



## 9. How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Launch the notebook:

   ```bash
   jupyter notebook churn_modeling.ipynb
   ```
3. Run all cells sequentially to reproduce the full workflow (EDA → modeling → evaluation → SHAP).
4. The final predictions are saved to:

   ```
   outputs/churn_predictions.csv
   ```



## 10. Future Work

* Extend model to time-to-churn prediction using survival analysis
* Integrate customer lifetime value forecasting
* Build an interactive dashboard for ROI and churn risk visualization
* Deploy model as an API for real-time scoring



## 11. References

1. IBM Telco Customer Churn Dataset, Kaggle
2. Verbeke, W. et al., “New insights into churn prediction in the telecommunication sector” (2012)
3. CatBoost, LightGBM, XGBoost official documentation
4. Optuna: A Next-generation Hyperparameter Optimization Framework



