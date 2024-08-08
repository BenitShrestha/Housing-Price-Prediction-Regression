from main import load_data, label_encoder, split_data, prediction_generation, load_pickle
from evaluation import evaluate_predictions, evaluate_classification
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Example usage
df_train, df_validation = load_data('housing_price_prediction/train.csv', 'housing_price_prediction/validation.csv')

df_train = label_encoder(df_train)
df_validation = label_encoder(df_validation)

X_train, y_train, X_valid, y_valid = split_data(df_train, df_validation)

# Drop the hotwaterheating feature from validation
X_valid = X_valid.drop('hotwaterheating', axis = 1)

model = load_pickle('Saved_Models/rf_model.pkl')

y_pred = prediction_generation(model, X_valid)

metrics = evaluate_predictions(y_valid, y_pred)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")