import requests
import json
import os

with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

#Call each API endpoint and store the response
response1 = requests.post('http://127.0.0.1:8000/prediction', json={'datasetfolder':'testdata','dataset':'testdata.csv'}).text
response2 = requests.get('http://127.0.0.1:8000/scoring', json={'output_model_path':output_model_path,'test_data_path':test_data_path, 'dataset': 'testdata.csv'}).text
response3 = requests.get('http://127.0.0.1:8000/summarystats').text
response4 = requests.get('http://127.0.0.1:8000/diagnostics').text


#combine all API responses
responses = 'prediction: '+response1+'\n'+'scoring: '+response2+'\nsummary stats: '+response3+'\ndiagnostics: '+response4
#write the responses to your workspace
with open('apireturns.txt', 'w') as f:
    f.write(responses)
f.close()


