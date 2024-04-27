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
    Download training/validation/testing data.
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
    Get Dataframes For Analysis.
    """
    df_list = []
    for i in range(1, 4):
        df_list.append(pd.read_csv(config.get('data', f'file{i}')))
    print(df_list)
    return df_list

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

@main.command('train_bn')
def train_nb():
    """
    Train a Bernoulli Naive Bayes Model
    """
    bnb = BernoulliNB()
    vec_1 = CountVectorizer(tokenizer=process_text)
    X = vec.fit_transform(train["Comment"])
    y = train["Result_Bin"]
    bnb.fit(X,y)
    y_pred = BNB.predict(validation["Stemmed"])
    y_val = validation["Result_Bin"]
    # Calculate F1
    f1 = f1_score(y_val, y_pred)
    print("F1 Score:", round(f1,3))


if __name__ == "__main__":
    sys.exit(main())
