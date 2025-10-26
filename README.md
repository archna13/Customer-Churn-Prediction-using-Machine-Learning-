# Customer-Churn-Prediction-using-Machine-Learning
🧠 Customer Conversion Prediction

This project focuses on predicting customer conversion likelihood (whether a customer converts or not) using machine learning classification models. The dataset comprises behavioral, demographic, and session-based features, and the analysis aims to identify factors that influence conversion rates.

📂 Project Structure
classification_set.ipynb       # Main Jupyter Notebook
classification_data.csv        # Dataset used in analysis

⚙️ Key Steps in the Notebook
1. Importing Libraries

The project uses the following Python libraries:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

2. Data Loading

The dataset is loaded using:

data = pd.read_csv("classification_data.csv")

3. Exploratory Data Analysis (EDA)

Checked data types, missing values, and unique values.

Visualized the target variable (has_converted) distribution.

Calculated conversion percentages.

Dropped irrelevant or redundant columns such as:

["geoNetwork_latitude","geoNetwork_longitude","last_visitId",
 "visitId_threshold","earliest_visit_id","earliest_visit_number",
 "youtube","time_earliest_visit","days_since_last_visit",
 "days_since_first_visit","earliest_source","earliest_medium",
 "earliest_keyword","earliest_isTrueDirect","target_date"]

4. Handling Skewness and Encoding

Detected skewed numerical features and visualized them using boxplots, histograms, and violin plots.

Applied encoding techniques for categorical variables (likely Label Encoding or One-Hot Encoding).

5. Model Building

Two models were built and compared:

Logistic Regression

Random Forest Classifier

6. Hyperparameter Tuning

Grid Search or manual tuning was applied to optimize model performance.

7. Model Evaluation

Metrics used: Accuracy, Confusion Matrix, and Classification Report.

Evaluated model performance and compared both algorithms.

📊 Results (Typical)
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	~XX%	XX	XX	XX
Random Forest	~XX%	XX	XX	XX

(Replace XX with your actual results if known.)

🧩 Insights

Identified key features that influence conversion (e.g., session count, time spent, device type).

Random Forest likely performed better due to its handling of nonlinear relationships.

🚀 How to Run the Project

Clone this repository:

git clone https://github.com/yourusername/customer-conversion-prediction.git
cd customer-conversion-prediction


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook classification_set.ipynb

🧾 Requirements

You can create a requirements.txt file containing:

pandas
numpy
matplotlib
seaborn
scikit-learn

🧠 Future Improvements

Add more advanced models (XGBoost, LightGBM).

Implement feature selection or dimensionality reduction.

Use cross-validation for more robust accuracy.

Deploy using Flask or Streamlit.
