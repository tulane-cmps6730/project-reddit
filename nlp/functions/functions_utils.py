#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
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


def cnn_process(document):

    document = document.split()
    
    for item in document:
        document = [re.sub(r'^\W+|\W+$', "", item) for item in document]
            
    document = [item.lower() for item in document]
    
    document = " ".join(document)

    document = document.tolist()
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(document)
    
    document = tokenizer.texts_to_sequences(document)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    maxlen = 100
    document = pad_sequences(document, padding='post', maxlen=maxlen)
    
    return document

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





    