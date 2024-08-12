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

def main():
    # Load train, validation data as dataframes
    df_train = load_data('housing_price_prediction/train.csv')
    df_test = load_data('housing_price_prediction/housing-test-set.csv')

    # Encode categorical features
    df_train = label_encoder(df_train)
    df_test = label_encoder(df_test)

    # Split into input features(X) and target features(y)
    X_train, y_train = split_data(df_train)
    X_test, y_test = split_data(df_test)

    # Drop the hotwaterheating feature from validation
    X_test = X_test.drop('hotwaterheating', axis = 1)

    # Load the trained model
    model = load_pickle('Saved_Models/rf_model_66.pkl')

    ''' Generate predictions based on the trained model '''
    y_pred = prediction_generation(model, X_test)

    ''' Evaluate the predictions '''
    metrics = evaluate_predictions(y_test, y_pred)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()