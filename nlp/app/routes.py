from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import clf_path

import pickle
import sys

bnb, vec_1, process_text = pickle.load(open(clf_path, 'rb'))
print('read bnb %s' % str(bnb))
print('read vec %s' % str(vec_1))
print('read process_text %s' % str(process_text))
labels = ['loss', 'win']

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
        updated_field = process_text([input_field])
		X = vec.transform([input_field])
		pred = bnb.predict(X)[0]
		proba = bnb.predict_proba(X)[0].max()
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=labels[pred], confidence='%.2f' % proba)
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
