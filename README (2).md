# 🧠 Customer Conversion Prediction

This project focuses on predicting **customer conversion likelihood** (whether a customer converts or not) using machine learning classification models. The dataset includes behavioral, demographic, and session-based features, and the analysis aims to uncover factors influencing conversion rates.

---

## 📂 Project Structure

```
classification_set.ipynb       # Main Jupyter Notebook
classification_data.csv        # Dataset used in analysis
```

---

## ⚙️ Key Steps in the Notebook

### 1. Importing Libraries
The project uses the following Python libraries:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```

### 2. Data Loading
```python
data = pd.read_csv("classification_data.csv")
```

### 3. Exploratory Data Analysis (EDA)
- Checked data types, missing values, and unique values.  
- Visualized the **target variable** (`has_converted`) distribution.  
- Dropped irrelevant or redundant columns:
  ```
  ["geoNetwork_latitude","geoNetwork_longitude","last_visitId",
   "visitId_threshold","earliest_visit_id","earliest_visit_number",
   "youtube","time_earliest_visit","days_since_last_visit",
   "days_since_first_visit","earliest_source","earliest_medium",
   "earliest_keyword","earliest_isTrueDirect","target_date"]
  ```

### 4. Handling Skewness and Encoding
- Visualized skewed numerical features with boxplots, histograms, and violin plots.  
- Applied encoding techniques for categorical variables.  

### 5. Model Building
- Logistic Regression  
- Random Forest Classifier  

### 6. Hyperparameter Tuning
Used **Grid Search** or manual tuning for performance optimization.

### 7. Model Evaluation
Metrics used: **Accuracy**, **Confusion Matrix**, and **Classification Report**.

---

## 📊 Results (Example)

| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|-----------|--------|-----------|
| Logistic Regression | ~XX% | XX | XX | XX |
| Random Forest | ~XX% | XX | XX | XX |

---

## 🧩 Insights
- Key factors affecting conversion include session counts, average time, and device type.  
- Random Forest generally performs better due to handling of nonlinear relationships.

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/customer-conversion-prediction.git
cd customer-conversion-prediction
pip install -r requirements.txt
jupyter notebook classification_set.ipynb
```

---

## 🧾 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🧠 Future Improvements
- Try advanced models (XGBoost, LightGBM)
- Apply feature selection or PCA
- Deploy model using Flask or Streamlit

---

## 👩‍💻 Author
**Your Name**  
📧 your.email@example.com  
🔗 [GitHub Profile Link]
