""" Main Script - Consists of all functions required """

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

""" 
    Returns dataframes for train and validation when proper path is provided
    Parameters - Path to train data and validation data 
    Example - df_train, df_validation = load_data('housing_price_prediction/df_train.csv', 'housing_price_prediction/df_validation.csv')
"""
def load_data(train_path, validation_path):
    df_train = pd.read_csv(train_path)
    df_validation = pd.read_csv(validation_path)

    return df_train, df_validation

"""
    Returns label encoded dataframe
    Parameters - Dataframe
    Example - df_train = label_encoder(df_train), df_validation = label_encoder(df_validation)
"""
def label_encoder(df):
    list_to_encode = [name for name in df.columns if df[name].dtype == 'object']
    label_encoder = LabelEncoder()
    for name in list_to_encode:
        df[name] = label_encoder.fit_transform(df[name])

    return df

"""  
    Plots histplot for each feature of given dataframe
    Parameters - Dataframe, Label
    Example - histplot_all(df_train, 'train'), histplot_all(df_validation, 'validation')
"""
def histplot_all(df, data):
    print(f"Histplots for {data} data:\n")

    for i in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[i], kde = True)
        plt.title(f"Histogram of {i}")
        plt.xlabel(i)
        plt.ylabel("Count")
        plt.show()

"""  
    Plots heatmap for each feature of given dataframe
    Parameters - Dataframe, Label
    Example - heatmap_all(df_train, 'train'), heatmap_all(df_validation, 'validation')
"""
def heatmap_all(df, data):
    print(f"Heatmap for {data} data:\n")

    plt.figure(figsize = (12, 8))
    sns.heatmap(df.corr(), annot = True)
    plt.show()

"""
    Splits dataframe into input features: X and target features: y
    Parameters - Dataframes
    Example - X_train, y_train, X_valid, y_valid = split_data(df_train, df_validation)
"""
def split_data(df_train, df_validation):
    X_train = df_train.drop("price", axis = 1)
    y_train = df_train['price']

    X_valid = df_validation.drop('price', axis =1)
    y_valid = df_validation['price']

    return X_train, y_train, X_valid, y_valid

"""
    Loads pickle model
    Parameters - Path to pickle file
    Example - model = load_pickle('Saved_Models/rf_model.pkl')
"""
def load_pickle(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model

"""
    Generates predictions
    Parameters - Model
    Example - y_pred = prediction_generation(model)
"""
def prediction_generation(model, X_valid):
    y_pred = model.predict(X_valid)

    return y_pred