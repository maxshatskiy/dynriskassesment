from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    file = open(os.path.join('practicemodels','trainedmodel.pkl'),'rb')
    model = pickle.load(file)
    test_data = pd.read_csv(os.path.join(test_data_path,"testdata.csv"))
    X_test = test_data[["lastmonth_activity","lastyear_activity","number_of_employees"]]
    y_test = test_data["exited"]
    y_pred = model.predict(X_test)
    f1score = metrics.f1_score(y_test, y_pred)

    outfile = open(os.path.join(test_data_path,'latestscore.txt'),'w')
    outfile.write(str(f1score))
    outfile.close()

    return f1score

if __name__=="__main__":

    score_model()


