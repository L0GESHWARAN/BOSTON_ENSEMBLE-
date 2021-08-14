from flask import Flask,request,render_template,jsonify
from flask_cors import CORS,cross_origin
import numpy
import pandas
import pickle

app = Flask(__name__)
@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])

            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = int(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO= float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])
            is_CHAS = request.form['CHAS']
            if (is_CHAS == 'YES'):
                CHAS = 1
            else:
                CHAS = 0
            file = "model.sav"
            file1 = 'scaler.sav'
            model=pickle.load(open(file,'rb'))
            scaler = pickle.load(open(file1,'rb'))
            inputs = scaler.fit_transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
            prediction = model.predict(inputs)

            return render_template('result.html',prediction=prediction)
        except Exception as e:
            print(e)
            return "SOMETHING WRONG"
            
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
