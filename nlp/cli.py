# -*- coding: utf-8 -*-

"""Text Classification Project"""
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
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW  
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


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
    Download the Data
    """

    print("Configuration file path:", config_path)
    
    for i in range(1, 4):  
        data_url = config.get('data', f'url{i}')
        data_file = config.get('data', f'file{i}')
        print('downloading from %s to %s' % (data_url, data_file))
        response = requests.get(data_url)
        with open(data_file, 'wb') as f: 
            f.write(response.content)
            

def process_text(document):

    tokens = document.split()
    tokens = [re.sub(r'^\W+|\W+$', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text


@main.command('nb')
def train_nb():
    """
    Get Naive Bayes Metrics
    """
    train_df = pd.read_csv(config.get('data', 'file1'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    train_df["Comment"] = train_df["Comment"].apply(process_text)
    bnb_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), binary=True)
    X_train = bnb_vectorizer.fit_transform(train_df["Comment"])
    y_train = train_df["Result_Bin"]
    test_df["Comment"] = test_df["Comment"].apply(process_text)
    X_test = bnb_vectorizer.transform(test_df["Comment"]) 
    y_test = test_df["Result_Bin"]

    # Training the model
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = bnb.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("F1 Score:", round(f1, 3))
    print("Precision:", round(precision,3))
    print("Recall:", round(recall, 3))
 
    pickle.dump((bnb, bnb_vectorizer), open(bnb_path, 'wb'))

@main.command('lr')
def train_lr():
    """
    Get Logistic Regression
    """

    train_df = pd.read_csv(config.get('data', 'file1'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    train_df["Comment"] = train_df["Comment"].apply(process_text)
    lr_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    X_train = lr_vectorizer.fit_transform(train_df["Comment"])
    y_train = train_df["Result_Bin"]
    test_df["Comment"] = test_df["Comment"].apply(process_text)
    X_test = lr_vectorizer.transform(test_df["Comment"]) 
    y_test = test_df["Result_Bin"]

    # Training the model
    lr = LogisticRegression(max_iter = 1000)
    lr.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = lr.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("F1 Score:", round(f1, 3))
    print("Precision:", round(precision,3))
    print("Recall:", round(recall, 3))
    pickle.dump((lr, lr_vectorizer), open(lr_path, 'wb'))

@main.command('cnn')
def train_cnn():
    
    
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    #model_path = '/Users/jackiecollopy/Downloads/project-reddit/nlp/cnn_model.h5'
    model_path = os.path.join('nlp', 'cnn_model.h5')
    model = load_model(model_path, compile=False)

    tokenizer = Tokenizer()

    texts = pd.concat([train_df["Comment_Adj"], val_df["Comment_Adj"], test_df["Comment_Adj"]])
    tokenizer.fit_on_texts(texts)
    all_sequences = tokenizer.texts_to_sequences(texts)

    maxlen = np.percentile([len(x) for x in all_sequences], 95)  # 95th percentile
    maxlen = int(maxlen)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    X_test_sequences = tokenizer.texts_to_sequences(test_df["Comment_Adj"])
    X_test = pad_sequences(X_test_sequences, padding='post', maxlen=maxlen)
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(test_df["Result_Bin"])
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int) 
    
    f1 = f1_score(y_test, predictions)
    print("F1 Score:", round(f1,3))
    precision = precision_score(y_test, predictions)
    print("Precision:", round(precision, 3))
    recall = recall_score(y_test, predictions)
    print("Recall:", round(recall, 3))
        
    pickle.dump(model, open(cnn_path, 'wb'))
    
@main.command('bert')
def train_bert():
    '''
    Get BERT Metrics
    '''
    train_df = pd.read_csv(config.get('data', 'file1'))
    val_df = pd.read_csv(config.get('data', 'file2'))
    test_df = pd.read_csv(config.get('data', 'file3'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('nlp', 'bert.pth')
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-mini')

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
        
    train_encodings = tokenize(train_df)
    #val_encodings = tokenize(val_df)
    test_encodings = tokenize(test_df)
    
    train_dataset = CommentsDataset(train_encodings, train_df['Result_Bin'])
    #val_dataset = CommentsDataset(val_encodings, val_df['Result_Bin'])
    test_dataset = CommentsDataset(test_encodings, test_df['Result_Bin'])
    
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=100)
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
    print("Precision:", round(precision,3))
    print("Recall:", round(recall,3))
    print("F1 Score:", round(f1,3))
    
if __name__ == "__main__":
    sys.exit(main())
