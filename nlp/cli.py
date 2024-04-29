# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys

import os
import numpy as np
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from nlp.functions.functions_utils import process_text, cnn_process, basic_process
from . import bnb_path, lr_path, cnn_path, lr_path, config, config_path

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=6331, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
@main.command('dl-data')
def dl_data():
    """
    Get data (Do First)
    """

    print("Configuration file path:", config_path)
    
    for i in range(1, 4):  # Ensure this matches the number of URLs in your configuration
        data_url = config.get('data', f'url{i}')
        data_file = config.get('data', f'file{i}')
        print('downloading from %s to %s' % (data_url, data_file))
        response = requests.get(data_url)
        with open(data_file, 'wb') as f:  # Use 'wb' for writing in binary mode which is suitable for files downloaded from URLs
            f.write(response.content)
            
@main.command('data2df')
def data2df():
    """
    Get Dataframes (Do Second)
    """
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    # Optionally print DataFrames to verify contents
    print("Train DataFrame:")
    print(train_df.head())
    print("Validation DataFrame:")
    print(val_df.head())
    print("Test DataFrame:")
    print(test_df.head())

    # Return the DataFrames as separate variables
    return train_df, val_df, test_df


def process_text(document):

    tokens = document.split()
    tokens = [re.sub(r'^\W+|\W+$', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem the tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a string
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text


@main.command('train_nb')
def train_nb():
    """
    Naive Bayes Model (Do Third)
    """
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))

    train_df["Comment"] = train_df["Comment"].apply(process_text)
    bnb_vectorizer = CountVectorizer()
    X_train = bnb_vectorizer.fit_transform(train_df["Comment"])
    y_train = train_df["Result_Bin"]
    val_df["Comment"] = val_df["Comment"].apply(process_text)
    X_val = bnb_vectorizer.transform(val_df["Comment"]) 
    y_val = val_df["Result_Bin"]

    # Training the model
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = bnb.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print("F1 Score:", round(f1, 3))
    pickle.dump((bnb, bnb_vectorizer), open(bnb_path, 'wb'))

@main.command('train_lr')
def train_lr():
    """
    Logistic Regression Model (Do Fourth)
    """

    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))

    train_df["Comment"] = train_df["Comment"].apply(process_text)
    lr_vectorizer = CountVectorizer()
    X_train = lr_vectorizer.fit_transform(train_df["Comment"])
    y_train = train_df["Result_Bin"]
    val_df["Comment"] = val_df["Comment"].apply(process_text)
    X_val = lr_vectorizer.transform(val_df["Comment"]) 
    y_val = val_df["Result_Bin"]

    # Training the model
    lr = LogisticRegression(max_iter = 1000)
    lr.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = lr.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print("F1 Score:", round(f1, 3))
    pickle.dump((lr, lr_vectorizer), open(lr_path, 'wb'))

@main.command('train_cnn')
def train_cnn():
    '''
    Get CNN Model
    '''
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    model_path = '/Users/jackiecollopy/Downloads/project-reddit/nlp/cnn_model.h5'
    tokenizer_path = '/Users/jackiecollopy/Downloads/project-reddit/nlp/tokenizer.pickle'  # Adjust path as needed

    try:
        
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        
    def process_with_tokenizer(text):
        return cnn_process(text, tokenizer, maxlen=100)

    X_train = train_df["Comment_Adj"].apply(process_with_tokenizer)
    X_val = val_df["Comment_Adj"].apply(process_with_tokenizer)
    predictions = cnn.predict(X_val)
    predictions = (predictions > 0.5).astype(int) 
    
    f1 = f1_score(y_val, predictions)
    print("F1 Score on Validation:", round(f1,3))
    # Calculate Precision
    precision = precision_score(y_val, predictions)
    print("Precision on Validation:", round(precision, 3))
    # Calculate recall
    recall = recall_score(y_val, predictions)
    print("Recall on Validation:", round(recall, 3))

    pickle.dump((cnn), open(cnn_path, 'wb'))


    
if __name__ == "__main__":
    sys.exit(main())
