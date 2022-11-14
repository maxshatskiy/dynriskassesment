
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(test_data_path, test_data):
    #read the deployed model and a test dataset, calculate predictions
    file = open(os.path.join('production_deployment','trainedmodel.pkl'),'rb')
    model = pickle.load(file)

    test_data = pd.read_csv(os.path.join(test_data_path,test_data))
    X_test = test_data[["lastmonth_activity","lastyear_activity","number_of_employees"]]
    y_pred = model.predict(X_test)

    return list(y_pred)

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.txt"), sep=",")
    nan = list(df.isna().sum()/df.shape[0])
    summary_numerical = df.select_dtypes(include=[np.number]).agg(['max','median','std'], axis=0)
    summary_category = df.select_dtypes(include=['category','object']).agg(['mode'], axis=0)
    summary_list = summary_numerical.values.tolist()+summary_category.values.tolist()+nan

    return summary_list


##################Function to get timings
def execution_time():

    def ingestion_time():
        starttime = timeit.default_timer()
        os.system('python3 ingestion.py')
        timing=timeit.default_timer() - starttime
        return timing

    def training_time():
        starttime = timeit.default_timer()
        os.system('python3 training.py')
        timing=timeit.default_timer() - starttime
        return timing

    return [ingestion_time(), training_time()]

##################Function to check dependencies
def outdated_packages_list():

    outdated = subprocess.check_output(['pip','list','--outdated'], text=True)
    with open('outdated.txt','w') as f:
        f.write(outdated)
    subprocess.check_output(['pip', 'list', '--outdated'])
    return outdated

if __name__ == '__main__':
    model_predictions(test_data_path, test_data = "testdata.csv")
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
