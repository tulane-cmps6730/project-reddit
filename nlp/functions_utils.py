#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

