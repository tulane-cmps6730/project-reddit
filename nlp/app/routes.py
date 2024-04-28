from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import bnb_path
from ..functions.function_utils import process_text

import pickle
import sys

bnb, vec_1 = pickle.load(open(bnb_path, 'rb'))
print('read bnb %s' % str(bnb))
print('read vec %s' % str(vec_1))
labels = ['loss', 'win']

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_field = form.input_field.data
        updated_field = process_text(input_field)  # Assuming process_text function is defined globally
        X = vec_1.transform([updated_field])
        pred = bnb.predict(X)[0]
        proba = bnb.predict_proba(X)[0].max()
        # flash(input_field)
        return render_template('myform.html', title='', form=form, 
                               prediction=labels[pred], confidence='%.2f' % (proba * 100))
    return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
