# NON-ESSENTIAL PART OF THE PROJECT
# EXAMPLE ONLY FOR RANDOM FOREST MODEL
import json
import pandas as pd
import requests

url = 'http://127.0.0.1:8005/model'
test = pd.read_csv('test.csv', index_col=['ind', 'aggregation_code', 'county_name'])
request_data = json.dumps(test[180:181].to_json())
response = requests.post(url, request_data)
print(response.text)
