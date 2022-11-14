

import training
import scoring
import deployment
import diagnostics
import reporting
import pandas as pd
import os
import json

with open('config.json','r') as f:
    config = json.load(f)

##################Check and read new data
#first, read ingestedfiles.txt
dataset_csv_path = os.path.join(config['input_folder_path'])
ingested_data_path = os.path.join(config['output_folder_path'])
test_dataset_location = os.path.join(config['test_data_path'])
prod_deployment_path = config['prod_deployment_path']

df = pd.read_csv(os.path.join(ingested_data_path,'ingestedfiles.txt'), header=None, sep='\t')
ingesteddata_list = list(df.loc[:,0])
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(dataset_csv_path)
csv_filenames = []
for file in filenames:
    if (".csv" in file) and (file not in ingesteddata_list):
        csv_filenames.append(file)
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not csv_filenames:
    print('Exit since no new files to ingest')
    exit()
else:
    pass
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
f1score_new_model = scoring.score_model(prod_deployment_path, test_data_path="ingesteddata", test_data="finaldata.txt")
with open(os.path.join(prod_deployment_path,'latestscore.txt'),'r') as f:
    lines = f.readline()
f.close()

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
print('old model score {}'.format(round(float(lines), 3)))
print('new model score {}'.format(round(f1score_new_model, 3)))

if round(float(lines), 3) == round(f1score_new_model, 3):
    print('Exit since no model drift was found')
    exit()
else:
    ##################Re-deployment

    deployment.store_model_into_pickle()
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python diagnostics.py')
reporting.model_predictions(test_dataset_location, test_data="testdata.csv")
os.system('python apicalls.py')







