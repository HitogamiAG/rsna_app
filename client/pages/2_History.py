import streamlit as st
import requests
import dotenv
from datetime import datetime as dt

config = dotenv.dotenv_values('.env')
f = '%Y-%m-%dT%H:%M:%S'

if st.button('Reload'):
    pass    

match_dict = {
        'Id' : 'ID',
        'User_id' : None,
        'Filename' : 'Filename',
        'Datetime' : 'Date',
        'State' : 'Status',
        'Vt1' : 'Vertebra 1',
        'Vt1_conf' : None,
        'Vt2' : 'Vertebra 2',
        'Vt2_conf' : None,
        'Vt3' : 'Vertebra 3',
        'Vt3_conf' : None,
        'Vt4' : 'Vertebra 4',
        'Vt4_conf' : None,
        'Vt5' : 'Vertebra 5',
        'Vt5_conf' : None,
        'Vt6' : 'Vertebra 6',
        'Vt6_conf' : None,
        'Vt7' : 'Vertebra 7',
        'Vt7_conf' : None,
        'pixel_spacing_w' : None,
        'pixel_spacing_h' : None,
        'slice_thickness' : None
    }

resp = requests.get(f'{config["SERVER_HOST"]}/history')
data = resp.json()['message']
data = sorted(data, key = lambda x: dt.strptime(x['datetime'].split('.')[0], f), reverse=True)
#print(data)

if len(data) == 0:
    st.text("Sorry, you didn't upload any files")
else:
    table_form = {k : [] for k in match_dict.values() if k}
    
    for row in data:
        for k, v in match_dict.items():
            if not v: continue;
            if 'Vertebra' in v:
                pred = row[k.lower()]
                prob = row[k.lower() + '_conf']
                
                if pred is not None:
                    prob = str(int(abs((prob - (1 - pred)) * 100))) + '%'
                    pred = 'Damaged' if pred else 'Safe'
                
                table_form[v].append(f'{pred} | {prob}')
            elif 'Date' in v:
                table_form[v].append(dt.strptime(row[k.lower()].split('.')[0], f))
            else:
                table_form[v].append(row[k.lower()])
    
    st.table(table_form)
