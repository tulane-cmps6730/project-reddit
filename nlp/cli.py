# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys

import os
import io
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
import tensorflow as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
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
    model = load_model(model_path, compile=False)

    tokenizer = Tokenizer()

    texts = pd.concat([train_df["Comment_Adj"], val_df["Comment_Adj"], test_df["Comment_Adj"]])
    tokenizer.fit_on_texts(texts)
    all_sequences = tokenizer.texts_to_sequences(texts)

    maxlen = np.percentile([len(x) for x in all_sequences], 95)  # 95th percentile
    maxlen = int(maxlen)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    X_val_sequences = tokenizer.texts_to_sequences(val_df["Comment_Adj"])
    X_val = pad_sequences(X_val_sequences, padding='post', maxlen=maxlen)
    label_encoder = LabelEncoder()
    y_val = label_encoder.fit_transform(val_df["Result_Bin"])
    predictions = model.predict(X_val)
    predictions = (predictions > 0.5).astype(int) 
    
    f1 = f1_score(y_val, predictions)
    print("F1 Score:", round(f1,3))
    # Calculate Precision
    precision = precision_score(y_val, predictions)
    print("Precision:", round(precision, 3))
    # Calculate recall
    recall = recall_score(y_val, predictions)
    print("Recall:", round(recall, 3))
        
    pickle.dump(model, open(cnn_path, 'wb'))
    
@main.command('train_bert')
def train_bert():
    '''
    Get BERT
    '''
    
    file_url = 'https://drive.google.com/file/d/1K26N14tCziLm97ie7YeBBK8d_fz9oIg3/view?usp=sharing'

    # Download the file using requests library
    response = requests.get(file_url)

    # Load the file into a buffer
    buffer = io.BytesIO(response.content)

    # Load the model from the buffer
    model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(data, max_length=87):
        return tokenizer(
            data["Comment_Adj"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    class CommentsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels)
    
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item
    
        def __len__(self):
            return len(self.labels)
        
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,  # Ensure this matches the setup when the model was first created
            state_dict=None  # We'll load the state_dict next
            )
    train_encodings = tokenize(train_df)
    val_encodings = tokenize(val_df)
    test_encodings = tokenize(test_df)
    
    train_dataset = CommentsDataset(train_encodings, train_df['Result_Bin'])
    val_dataset = CommentsDataset(val_encodings, val_df['Result_Bin'])
    test_dataset = CommentsDataset(test_encodings, test_df['Result_Bin'])
    
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100)
    test_loader = DataLoader(test_dataset, batch_size=100)
    

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            true_labels.extend(batch['labels'].tolist())
    
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print("Precision on Test:", precision)
    print("Recall on Test:", recall)
    print("F1 Score on Test:", f1)
    
if __name__ == "__main__":
    sys.exit(main())
