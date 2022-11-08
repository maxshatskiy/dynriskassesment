from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    file = open(os.path.join('practicemodels','trainedmodel.pkl'),'rb')
    model = pickle.load(file)

    dest = os.path.join(prod_deployment_path,'latestscore.txt')
    src = os.path.join('testdata','latestscore.txt')
    open(dest, 'w').write(open(src, 'r').read())


    dest = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    src = os.path.join('ingesteddata', 'ingestedfiles.txt')
    open(dest, 'w').write(open(src, 'r').read())
        

    #model = pickle.load(open(os.path.join('practicemodels','trainedmodel.pkl'),'rb'))
    pickle.dump(model, open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'wb'))

if __name__=="__main__":

    store_model_into_pickle()