#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    # Return the processed text
    return ' '.join(stemmed_tokens)


tokenizer = Tokenizer()
train_df = pd.read_csv("/Users/jackiecollopy/Downloads/project-reddit/data/train.csv")
val_df = pd.read_csv("/Users/jackiecollopy/Downloads/project-reddit/data/train.csv")
test_df = pd.read_csv("/Users/jackiecollopy/Downloads/project-reddit/data/train.csv")

def basic_process(document):
    # Tokenize the document
    tokens = document.split()
    # Remove punctuation at the start and end of each token and convert to lowercase
    tokens = [re.sub(r'^\W+|\W+$', '', token).lower() for token in tokens]
    # Join processed tokens back into a string
    processed_text = ' '.join(tokens)
    return processed_text

def cnn_process(document):
    
    processed_document = basic_process(document)
    tokenizer = Tokenizer()

    texts = pd.concat([train_df["Comment_Adj"], val_df["Comment_Adj"], test_df["Comment_Adj"]])
    tokenizer.fit_on_texts(texts)

    all_sequences = tokenizer.texts_to_sequences(texts)
    sequences = tokenizer.texts_to_sequences([processed_document])
    
    padded_sequences = pad_sequences(sequences, maxlen=87, padding='post')
    return padded_sequences


def bert_process(document):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='tf'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask





    