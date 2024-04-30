from flask import render_template, flash, redirect, session
from tensorflow.keras.models import load_model
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW  
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from . import app
from .forms import MyForm
from .. import bnb_path, lr_path, cnn_path
from ..functions.functions_utils import process_text, basic_process, cnn_process

import pickle
import sys

bnb, bnb_vectorizer = pickle.load(open(bnb_path, 'rb'))
lr, lr_vectorizer = pickle.load(open(lr_path, 'rb'))
cnn = pickle.load(open(lr_path, 'rb'))


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_field = form.input_field.data
        model_choice = form.model_choice.data
        prediction = None
        proba = None 
        if model_choice == 'bnb':
            
            model = bnb
            text = process_text(input_field)
            text = bnb_vectorizer.transform([text])
            probas = bnb.predict_proba(text)
            positive_proba = probas[:, 1]
            if positive_proba > 0.5:
                prediction = "WIN"  
                proba = positive_proba
            else:
                prediction = "LOSS"  
                proba = 1 - positive_proba
        
        
        elif model_choice == 'lr':
            
            model = lr
            text = process_text(input_field)
            text = lr_vectorizer.transform([text])
            probas = lr.predict_proba(text)
            positive_proba = probas[:, 1]
            
            if positive_proba > 0.5:
                prediction = "WIN"
                proba = positive_proba
            else:
                prediction = "LOSS"
                proba = 1 - positive_proba
        
        elif model_choice == 'cnn':
            
            # For CNN, assuming preprocessing is handled differently or is built-in
            model_path = '/Users/jackiecollopy/Downloads/project-reddit/nlp/cnn_model.h5'
            model = load_model(model_path, compile=False)
            text = basic_process(input_field)
            text = cnn_process(text)
            probas = model.predict(text)
            preds = (probas > 0.5).astype(int) 
            if preds == 1:
                prediction = "WIN"
                proba = probas
            else:
                prediction = "LOSS"
                proba = 1 - probas
        elif model_choice == 'bert':
           
            tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-mini')
            model = AutoModelForSequenceClassification.from_pretrained('/Users/jackiecollopy/Downloads/project-reddit/notebooks/bert.pth')
            text = tokenizer(input_field, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**text)
            predicted_class = torch.argmax(outputs.logits).item()
            predicted_probability = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
            
            if predicted_class == 1:
                prediction = "WIN"
                proba = predicted_probability
            else:
                prediction = "LOSS"
                proba = 1 - predicted_probability

        return render_template('myform.html', title='', form=form, 
                               prediction=prediction, confidence='%.2f' % (proba * 100))
    return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
