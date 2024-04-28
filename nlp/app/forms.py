from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import DataRequired

class MyForm(FlaskForm):
	class Meta:  # Ignoring CSRF security feature.
		csrf = False

	input_field = StringField(label='input comment:', id='input_field',
							  validators=[DataRequired()], 
							  render_kw={'style': 'width:50%'})
    model_choice = RadioField('Choose Model', choices=[
    ('bnb', 'Naive Bayes'),
    ('lr', 'Logistic Regression'),
    ('cnn', 'Convolutional Neural Network')
    ])
	submit = SubmitField('Submit')