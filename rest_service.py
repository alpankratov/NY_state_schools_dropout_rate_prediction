# NON-ESSENTIAL PART OF THE PROJECT
# EXAMPLE ONLY FOR RANDOM FOREST MODEL
# needs to be run in prompt as 'python rest_service.py'
import pandas as pd
from flask import Flask, request
import pickle

local_model = pickle.load(open('models/rf.pickle', 'rb'))

app = Flask(__name__)

@app.route('/model',methods=['POST'])

def hello_world():
    request_data = pd.read_json(request.get_json(force=True))
    print(request_data.index)
    pred_proba = local_model.predict(request_data)
    return f"For item {request_data.index[0]} the probability is {round(pred_proba[0]*100,2)}%"

if __name__ == "__main__":
    app.run(port=8005, debug=True)