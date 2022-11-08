import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(input_folder_path)
    csv_filenames=[]
    for file in filenames:
        if ".csv" in file:
            csv_filenames.append(file)

    date_time = datetime.now()
    date_time_str = str(date_time.year)+'/'+str(date_time.month)+ '/'+str(date_time.day)
    #information_about_data = [input_folder_path, *csv_filenames, date_time_str]

    output_file = open(os.path.join(output_folder_path,"ingestedfiles.txt"),'w')
    for item in csv_filenames:
        output_file.write(str(item)+"\n")
    output_file.close()


    merged_df = pd.concat((pd.read_csv(os.path.join(input_folder_path,f)) for f in csv_filenames))
    merged_df.drop_duplicates(inplace=True)
    merged_df.to_csv(os.path.join(output_folder_path,"finaldata.txt"), index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
