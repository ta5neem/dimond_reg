# Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# from jinja2.utils import escape
# the function I craeted to process the data in utils.py
from Untitled6 import preprocess_new


# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
model = joblib.load('model3_XGBoost.pkl')

# Route for Home page


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        Unnamed = float(request.form['Unnamed:0'])
        carat = float(request.form['carat'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])


        # Remmber the Feature Engineering we did
        size = x*y*z


        # Concatenate all Inputs
        X_new = pd.DataFrame({'Unnamed:0': [Unnamed], 'carat': [carat], 'cut': [cut], 'color':[color],
                              'clarity': [clarity], 'depth': [depth], 'table': [table], 'x':[x],
                              'y': [y], 'z': [z], 
                              'size': [size]
                              })

        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        return render_template('predict.html', price_val=y_pred_new)
    else:
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)