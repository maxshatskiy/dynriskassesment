import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

#Call each API endpoint and store the response
response1 = requests.post('http://127.0.0.1:8000/prediction', json={'datasetlocation':'testdata'}).text
response2 = requests.get('http://127.0.0.1:8000/scoring').text
response3 = requests.get('http://127.0.0.1:8000/summarystats').text
response4 = requests.get('http://127.0.0.1:8000/diagnostics').text


#combine all API responses
responses = 'prediction: '+response1+'\n'+'scoring: '+response2+'\nsummary stats: '+response3+'\ndiagnostics: '+response4
#write the responses to your workspace
with open('apireturns.txt', 'w') as f:
    f.write(responses)
f.close()


