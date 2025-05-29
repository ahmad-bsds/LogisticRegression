# 📈 Logistic Regression for Marketing Campaign Prediction

This project implements a full pipeline for predicting the success of marketing campaigns using **Logistic Regression**. The dataset comes from a real-world marketing campaign and includes client demographics, account information, and contact history. The objective is to predict whether a client will subscribe to a term deposit (`y`: yes/no).

---

## 🔍 Project Overview

* **Dataset Size**: 45,211 records
* **Target Variable**: `y` (binary classification: `yes` → 1, `no` → 0)
* **Model Used**: Logistic Regression
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC Curve, AUC
* **Final Accuracy**: `99.15%`
* **AUC Score**: `0.9969`

---

## 📁 Dataset Description

Each record includes the following features:

* **Client Info**: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`
* **Contact Info**: `contact`, `day`, `month`, `duration`
* **Campaign Info**: `campaign`, `pdays`, `previous`, `poutcome`
* **Target**: `y` (binary response: `yes`/`no`)

---

## 🧪 Steps Performed

### 1. **Exploratory Data Analysis (EDA)**

* Data overview with `.head()`, `.info()`, and value counts for categorical features.
* Checked for missing values (none found).
* Analyzed class imbalance in the target variable (`yes`: \~11.7%).

### 2. **Assumption Checks for Logistic Regression**

* ✅ Binary Target Variable
* ✅ Independence of Observations
* ✅ No Severe Multicollinearity
  Used **Variance Inflation Factor (VIF)**. Highest VIF ≈ 5.
* ✅ No Extreme Outliers
  Used **Cook’s Distance** to identify and remove 4,775 influential data points.

### 3. **Data Preprocessing**

* Categorical columns encoded using **OneHotEncoder**
* Binary columns (`default`, `housing`, `loan`) converted to `0/1`
* Numerical columns scaled using **StandardScaler**
* All preprocessing handled via **scikit-learn Pipeline**

### 4. **Model Training**

* Data split: 70% training, 30% testing
* Trained using `LogisticRegression` with default `lbfgs` solver
* Handled convergence warning by noting iteration limits (could be improved with `max_iter`)

### 5. **Model Evaluation**

```plaintext
Accuracy: 99.15%

Classification Report:
               precision    recall  f1-score   support
         0       1.00      0.99      1.00     11617
         1       0.85      0.95      0.90       483

ROC AUC Score: 0.9969
```

---

## 📊 Visualizations

* **Boxplots** for detecting outliers in numerical features
* **Cook’s Distance Plot** for influential data points
* **ROC Curve** to evaluate classification thresholds

---

## 📦 Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

1. Clone the repo:

```bash
git clone https://github.com/ahmad-bsds/LogisticRegression.git
cd LogisticRegression
```

2. Open the Jupyter notebook:

```bash
jupyter notebook LogisticRegressionMarketing.ipynb
```

---

## 📌 Key Learnings

* Logistic Regression assumptions and diagnostics
* Detecting and handling multicollinearity and outliers
* Modular data preprocessing using scikit-learn Pipelines
* Evaluating classification models using AUC and ROC

---

## 🤝 Contributions

Feel free to open issues or pull requests for improvements, feature additions, or model enhancements.
