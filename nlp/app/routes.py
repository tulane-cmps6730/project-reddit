from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import bnb_path, lr_path, cnn_path
from ..functions.functions_utils import process_text, basic_process, cnn_process

import pickle
import sys

bnb, bnb_vectorizer = pickle.load(open(bnb_path, 'rb'))
lr, lr_vectorizer = pickle.load(open(lr_path, 'rb'))
cnn = pickle.load(open(lr_path, 'rb'))


labels = ['loss', 'win']

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_field = form.input_field.data
        model_choice = form.model_choice.data
        
        if model_choice == 'bnb':
            model = bnb
            text = process_text(input_field)
            text = bnb_vectorizer.transform([text])
            pred_labels = bnb.predict(text)
            probas = bnb.predict_proba(text)
            pred = pred_labels[0]
            proba = probas[0, pred]
        elif model_choice == 'lr':
            model = lr
            text = process_text(input_field)
            text = lr_vectorizer.transform([text])
            pred_labels = lr.predict(text)
            probas = lr.predict_proba(text)
            pred = pred_labels[0]
            proba = probas[0, pred]
        elif model_choice == 'cnn':
            # For CNN, assuming preprocessing is handled differently or is built-in
            text = basic_process(input_field)
            text = cnn_process(text)
            predictions_proba = cnn.predict([text])
            pred = (predictions_proba > 0.5).astype(int)[0]  # Assuming binary classification
            proba = predictions_proba[0]
        elif model_choice == 'bert':
            tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-mini')
            def tokenize(data, max_length=87):
            return tokenizer(
                data["Comment_Adj"].tolist(),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            text = tokenize([input_field])

        return render_template('myform.html', title='', form=form, 
                               prediction=labels[pred], confidence='%.2f' % (proba * 100))
    return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
