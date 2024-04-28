# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys

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
from . import clf_path, config, config_path

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
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



@main.command('train_nb')
def train_nb():
    """
    Naive Bayes Model (Do Third)
    """
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))

    def process_text(document):
    # Tokenize the document
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
    
    bnb = BernoulliNB()
    vec_1 = CountVectorizer(tokenizer=process_text)
    X = vec_1.fit_transform(train_df["Comment"])
    y = train_df["Result_Bin"]
    bnb.fit(X,y)
    y_pred = nb.predict(val_df["Stemmed"])
    y_val = val_df["Result_Bin"]
    # Calculate F1
    f1 = f1_score(y_val, y_pred)
    print("F1 Score:", round(f1,3))

@main.command('train_lr')
def train_lr():
    """
    Logistic Regression Model (Do Fourth)
    """

    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))

    def process_text(document):
    # Tokenize the document
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
        
    lr = LogisticRegression()
    vec_2 = CountVectorizer(tokenizer=process_text)
    X = vec_2.fit_transform(train_df["Comment"])
    y = train_df["Result_Bin"]
    lr.fit(X,y)
    y_pred = lr.predict(val_df["Stemmed"])
    y_val = val_df["Result_Bin"]
    # Calculate F1
    f1 = f1_score(y_val, y_pred)
    print("F1 Score:", round(f1,3))


if __name__ == "__main__":
    sys.exit(main())
