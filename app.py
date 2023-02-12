import pickle
import numpy as np
from flask import Flask,request,jsonify,app,url_for,render_template
# import pandas as pd

# starting flask application
app = Flask(__name__)

# loading by unpickling the files
regmodel = pickle.load(open('LinearReg.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

# home page
@app.route('/')
def home():
    return render_template('home.html')

# predict api which we test in postman 
@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    # print(type(data))
    # print(data.values())
    # print(list(data.values()))
    print(np.array(list(data.values())).reshape(1,-1))

    # converting the json data to the format of scalar input so we can give it to prediction
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)