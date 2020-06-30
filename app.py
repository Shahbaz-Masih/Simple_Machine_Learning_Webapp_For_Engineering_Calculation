import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

lrModel = pickle.load(open('lrModel.pkl', 'rb'))
dtModel = pickle.load(open('dtModel.pkl', 'rb'))
rfModel = pickle.load(open('rfModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictOutput',methods=['POST'])
def Outpredictput():
    '''
    For rendering results on HTML GUI
    Reading the entered parameters and saving them in a list
    '''

    at = request.form["AT"]
    v = request.form["V"]
    ap = request.form["AP"]
    rh = request.form["RH"]

    parameters = [at, v, ap, rh]
# changing the input to float
    float_features = [float(x) for x in parameters]
# changing the list to numpy array
    final_features = [np.array(float_features)]
# Taking the input of the model type and selecting the right model for predicting power output
    modeltype = request.form["models"]

    if modeltype == "Linear":
        prediction = lrModel.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text='With Linear Regression Model, Plant Power Output is {} MW'.format(output))

    if modeltype == "DecisionTree":
        prediction = dtModel.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text='With Decision Tree Regressor, Plant Power Output is {} MW'.format(output))

    if modeltype == "RandomForest":
        prediction = rfModel.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text='With Random Forest Regressor, Plant Power Output is {} MW'.format(output))


if __name__ == "__main__":
    app.run(debug=True)