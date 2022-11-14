from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions
from scoring import score_model
from diagnostics import dataframe_summary
from diagnostics import execution_time
from diagnostics import outdated_packages_list



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    dataset_location = request.get_json(force=True)
    #y_pred = model_predictions(dataset_location['datasetlocation'])
    y_pred = model_predictions(dataset_location['datasetfolder'], dataset_location['dataset'])
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    parameters = request.get_json(force=True)
    #f1score = score_model()
    f1score = score_model(model_path = parameters['output_model_path'], test_data_path = parameters['test_data_path'], test_data=parameters['dataset'])
    return str(f1score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    summary = dataframe_summary()
    return list(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():

    ext_time = execution_time()
    outdated_packages = outdated_packages_list()
    return 'execution time [ingestion, training]: '+str(ext_time) + '\noutdated packages: \n' + outdated_packages

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
