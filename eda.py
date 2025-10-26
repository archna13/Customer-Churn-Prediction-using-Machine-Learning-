import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load a sample classification dataset
from sklearn.datasets import load_iris

df=pd.read_csv(r"C:\Users\shali\Desktop\FINAL PROJECT\classification_data.csv")

# Streamlit app
def main():
    st.title("Remove Column and Save File")

    # Display the original DataFrame
    st.write("## Original Data:")
    st.write(df)

    # Choose a column to remove
    column_to_remove = st.selectbox("Select a column to remove:", df.columns)

    # Remove the selected column
    df_removed_column = df.drop(columns=[column_to_remove])

    # Display the modified DataFrame
    st.write("## Modified Data (Column Removed):")
    st.write(df_removed_column)

    # Save the modified DataFrame to a new file
    save_button = st.button("Save Modified Data")

    if save_button:
        save_path = 'modified_data.csv'  # Replace with your desired save path
        df_removed_column.to_csv(save_path, index=False)
        st.success(f"Modified data saved to {save_path}")

if __name__ == "__main__":
    main()
df=pd.read_csv("modified_data.csv")    
# Streamlit app
def main():
    st.title("Classification EDA with Streamlit")

    # Display dataset
    st.write("## Dataset Overview:")
    st.write(df.head())

    # Summary statistics
    st.write("## Summary Statistics:")
    st.write(df.describe())

    # target distribution
    st.write("## Target Distribution:")
    class_counts = df['has_converted'].value_counts()
    st.bar_chart(class_counts)

    """# Pairplot
    st.write("## Pairplot:")
    sns.pairplot(df, hue='has_converted')
    st.pyplot()"""
    # Categorical Variables
    cat_cols=df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    p_d=pd.DataFrame(num_cols)
    # Correlation matrix heatmap
    st.write("## Correlation Matrix Heatmap:")
    
    sns.heatmap(p_d.corr(), annot=True, cmap='coolwarm')
    st.pyplot()
    # Assuming 'categorical_feature' is a categorical column in your DataFrame
    st.write("## Count Plot for Categorical Feature:")
    sns.countplot(x='categorical_feature', data=num_cols)
    st.pyplot()
    ## Violin Plot for Feature Distributions
    st.write("## Violin Plot for Feature Distributions:")
    for feature in iris.feature_names:
        sns.violinplot(x='target', y=feature, data=df)
        st.pyplot()

    # Box plot for each feature by class
    st.write("## Box Plots by Class:")
    for feature in iris.feature_names:
        sns.boxplot(x='target', y=feature, data=df)
        st.pyplot()

if __name__ == "__main__":
    main()