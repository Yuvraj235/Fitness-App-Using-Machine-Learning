from flask import Flask, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from tensorflow.keras.models import load_model
import numpy as np
import joblib

def return_prediction(model,scaler,sample_json):
    
    gen = sample_json['Gender']
    hei = sample_json['Height']
    wei = sample_json['Weight']
    
    person = [[gen,hei,wei]]
    
    classes = np.array(['Extremely Weak','Weak','Normal','Overweight','Obesity','Extreme Obesity'])
    
    person = scaler.transform(person)
    
    class_in = model.predict_classes(person)[0]
    
    return classes[class_in]



app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class PersonForm(FlaskForm):

	gen_p = TextField("Gender(Male:0/Female:1) : ")
	hei_p = TextField("Height(Cm) : ")
	wei_p = TextField("Weight(Kg) : ")

	submit = SubmitField("submit")

@app.route("/",methods=['GET', 'POST'])
def index():

	form = PersonForm()

	if form.validate_on_submit():

		session['gen_p'] = form.gen_p.data
		session['hei_p'] = form.hei_p.data
		session['wei_p'] = form.wei_p.data

		return redirect(url_for("prediction"))


	return render_template('home.html', form=form)

he_we_model = load_model('final_he_we.h5')
he_we_scaler= joblib.load('he_we_scaler.pkl')

@app.route('/prediction')
def prediction():

	content = {}

	content['Gender'] = int(session['gen_p'])
	content['Height'] = int(session['hei_p'])
	content['Weight'] = int(session['wei_p'])

	results = return_prediction(model=he_we_model,scaler=he_we_scaler,sample_json=content)

	return render_template('prediction.html',results=results)


if __name__=='__main__':
	app.run(debug=True, use_reloader=True)