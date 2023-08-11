import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from pandas import DataFrame
import json

# Json tiktok

import json
f = open('/content/drive/MyDrive/my_data/data_json/tiktok.json')
data_1 = json.load(f)['comments']

for i in data_1[:3]:
  print(i["cid"])
  
for i in range(0,len(data_1)):
    print(data_1[i])

data_new = pd.DataFrame()
df = pd.DataFrame(data_1)
pd.set_option('display.max_columns', None)
print(df.head())

df["uid"] = df["user"].apply(lambda x: x["uid"])
df["nickname"] = df["user"].apply(lambda x: x["nickname"])
df["avatar_uri"] = df["user"].apply(lambda x: x["avatar_uri"])



# Json facebook 




# d = open('/content/drive/MyDrive/my_data/data_json/data.json')
# data_d = json.load(d)
# data_d_new = pd.DataFrame(data_d)
# data_d_new


with open('/content/drive/MyDrive/my_data/data_json/data.json', 'r') as f:
    data = json.load(f)
    


data_new = []
for conversation in data['conversations']['data']:  # vòng lặp for lấy qua các conversation
    for message in conversation['messages']['data']: # vòng lặp for lấy qua các message
        data_new.append({
            'conversation_id': conversation['id'],
            'message_id': message['id'],
            'create_time': message['created_time'],
            'from_name': message['from']['name'],
            'from_email': message['from']['email'],
            'from_id': message['from']['id'],
            'to_name': message['to']['data'][0]['name'],
            'to_email': message['to']['data'][0]['email'],
            'to_id': message['to']['data'][0]['id'],
            'messenger': message['message']
        })
df = pd.DataFrame.from_dict(data_new, orient='columns')

print(df)


# 2 Json trong 1 data


t = open('/content/drive/MyDrive/my_data/data_json/data_NER_tokenized_pre_remove_empty_sentence.json')
data_sp = json.load(t)

df_sp = []
for record in data_sp:
  for entity in record['entities']:
    df_sp.append({
        'text' : record['text'],
        'intent' : record['intent'],
        'entity' : entity['entity'],
        'value': entity['value'],
        'start' : entity['start'],
        'end' : entity['end'],
        'language' : record['metadata']['language']
    })
    
df_final = pd.DataFrame.from_dict(df_sp,orient='columns')

print(df_final)

df_final['entity'].unique()

# xử lý tách dấu #

def custom_split(i):
    if len(i) > 8:
        split_values = i.split('#')
        return split_values[1] if len(split_values) > 1 else i
    return i

df_final['test'] = df_final['entity'].apply(custom_split)
df_final['test']