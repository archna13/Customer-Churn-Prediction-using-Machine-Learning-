## Customer Conversion Prediction using Machine Learning


### Introduction
This project using machine learning approach used to identify whether a customer is likely to complete a desired action based on their behavior and interactions. Analyzes customer behavioral, demographic, and session-based data to understand the factors that influence conversion. By applying classification algorithms, the system predicts customer conversion outcomes and compares model performance using relevant evaluation metrics. The insights can help businesses better understand customer behavior and support data-driven marketing and decision-making.


### Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook


### Installation

#### Clone the Repository

```bash
git clone https://github.com/archna13/Customer-Conversion-Prediction-using-Machine-Learning.git
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Jupyter Notebook

```bash
jupyter notebook classification_set.ipynb
```

### Features

* **Data Cleaning and Preprocessing:** Irrelevant and redundant columns that do not contribute meaningfully to conversion prediction are removed from the dataset. This reduces unnecessary information and prepares the data for model development.

* **Exploratory Data Analysis (EDA):** Numerical and categorical features are explored using statistical analysis and visualizations such as histograms, boxplots, violin plots, and target distribution plots. The analysis helps identify patterns, feature distributions, and factors related to customer conversion.

* **Feature Preprocessing & Encoding:** Categorical variables are transformed into numerical representations using encoding techniques, while numerical features are processed to handle different scales. Standardization is applied where required to prepare the features for machine learning models.

* **Model Building:** The processed dataset is divided into training and testing sets and used to build classification models. **Logistic Regression** is applied as a linear classification approach, while **Random Forest Classifier** is used to capture more complex and nonlinear relationships within customer behavior.

* **Hyperparameter Tuning:** Grid Search or manual parameter tuning is used to identify suitable model configurations and improve classification performance. This helps optimize the models before evaluating their final predictions.

* **Model Evaluation & Insights:** The trained models are evaluated using accuracy, confusion matrix, precision, recall, and F1-score through classification reports. The analysis indicates that customer session behavior, average time, and device-related attributes can contribute to conversion prediction, with Random Forest generally providing better performance.

### Results

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |      90% |      0.89 |   0.89 |     0.89 |
| Random Forest       |      93% |      0.90 |   0.90 |     0.90 |

* **Best Performing Model:** Random Forest achieved the highest reported accuracy of **93%**.
* **Logistic Regression:** Achieved a reported accuracy of **90%** with balanced precision, recall, and F1-score.
* **Model Comparison:** Random Forest generally performs better because it can capture nonlinear relationships between customer behavior and conversion outcomes.
* **Conversion Insights:** Session activity, average time, and device type are identified as important factors influencing customer conversion behavior.

### Future Enhancements

* **Advanced Models:** Experiment with XGBoost and LightGBM to improve prediction performance.
* **Feature Selection:** Apply feature selection techniques to identify the most influential conversion-related features.
* **Dimensionality Reduction:** Explore PCA to reduce feature dimensionality while retaining important information.
* **Web Deployment:** Deploy the trained model using Streamlit or Flask for real-time customer conversion prediction.

### Conclusion

This project demonstrates how machine learning can be used to predict customer conversion based on behavioral, demographic, and session-based data. The comparison of Logistic Regression and Random Forest shows that Random Forest provides stronger overall predictive performance for this dataset. The analysis also provides useful insights into customer behavior and the factors associated with conversion.

