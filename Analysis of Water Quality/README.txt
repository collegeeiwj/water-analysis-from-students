Water Potability Prediction
This script predicts water potability (whether water is drinkable or not) using Logistic Regression on a water quality dataset.

Key Steps:
Load & Clean Data: Reads the dataset and replaces missing values with the median.
Standardize Features: Normalizes the data for PCA analysis.
PCA: Reduces the dimensionality of the data.
Model Training: Trains a Logistic Regression model with various training-test splits.
Evaluation: Displays confusion matrices and classification reports.
Requirements:
numpy, pandas, matplotlib, seaborn, scikit-learn

Place the dataset water_potability.csv in the ./data/ folder.