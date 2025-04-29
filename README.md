# LogisticRegression

# Breast Cancer Prediction Using Logistic Regression
# Project Overview

This project is about building a machine learning model that can predict whether a tumor is benign (non-cancerous) or malignant (cancerous) based on different features of the tumor. We use Logistic Regression, a simple yet effective algorithm for binary classification problems. The goal is to help predict breast cancer based on the dataset provided.

# Dataset
The dataset used in this project contains information about tumors. The features (columns) in the dataset are related to the size, texture, and other properties of the tumor. The target variable is the diagnosis of the tumor, where:

B: Benign (non-cancerous)

M: Malignant (cancerous)

# Example Features:
radius_mean: Mean radius of the tumor

texture_mean: Mean texture of the tumor

perimeter_mean: Mean perimeter of the tumor

# Target:
# Diagnosis: Whether the tumor is benign or malignant.

# Steps to Run
# Prepare the Data:

The dataset is cleaned by removing irrelevant columns and handling missing values.

# Model Training:

The dataset is divided into two parts: training data and test data.

We train the Logistic Regression model on the training data.

# Model Evaluation:

After training, we evaluate the model’s performance using metrics like accuracy, precision, recall, and ROC-AUC. These help us understand how well the model is predicting tumor types.

# Outlier Detection:

We also check for outliers (unusual data points that may not be representative of normal tumor data) and remove them to improve the model’s performance.

# Key Concepts
# Logistic Regression: 
A method for predicting binary outcomes (e.g., benign vs. malignant).

# Precision and Recall: 
These are measures that help us understand how well the model is classifying tumors. Precision tells us how many of the tumors predicted as malignant are actually malignant. Recall tells us how many of the actual malignant tumors the model is able to detect.

# ROC-AUC: 
A curve used to measure the ability of the model to distinguish between benign and malignant tumors. A higher AUC value indicates a better model.

# How to Run the Project
# Install Python and Libraries:

You’ll need Python and the following libraries:

* Pandas

* NumPy

* Scikit-learn

* Matplotlib

* Seaborn

You can install these libraries by running:
----------------------------------------------------- -
pip install pandas numpy scikit-learn matplotlib seaborn
------------------------------------------------------- -
# Load the Dataset:

The dataset is loaded, cleaned, and preprocessed.

# Train the Model:

The Logistic Regression model is trained using the cleaned data.

# Evaluate the Model:

After training, we check how well the model performs by comparing predicted results with actual results.

# Results
After evaluating the model, we get a good understanding of how accurately it can predict if a tumor is benign or malignant. We use a confusion matrix, precision, recall, and ROC-AUC to measure the performance.

# Conclusion
This project demonstrates how to use Logistic Regression to classify breast cancer tumors based on various features. By applying different techniques like outlier detection and model evaluation, we can make the model more accurate and reliable for real-world predictions.

