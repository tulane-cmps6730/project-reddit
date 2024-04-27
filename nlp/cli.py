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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from . import clf_path, config

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
    
    for i in range(1, 4):
        data_url = config.get('data', f'url{i}')
        data_file = config.get('data', f'file{i}')
        print('downloading from %s to %s' % (data_url, data_file))
        r = requests.get(data_url)
        with open(data_file, 'wt') as f:
            f.write(r.text)
    
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

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()
    print('%d rows' % len(df))
    print('label counts:')
    print(df.partisan.value_counts())    



if __name__ == "__main__":
    sys.exit(main())
