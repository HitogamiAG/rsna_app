import streamlit as st
import requests
import base64
import time
import dotenv

config = dotenv.dotenv_values('.env')

uploaded_file = st.file_uploader("Choose a file: *")
#upload_button = st.button('Upload!')

if uploaded_file is not None and st.button('Upload!'):
    status = st.text("We are uploading your file... Please, wait")
    filename = uploaded_file.name
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    encoded_data = base64.b64encode(bytes_data)
    resp = requests.post(f'{config["SERVER_HOST"]}/upload', data={'filename' : filename, 'filedata' : encoded_data})
    status.text(resp.json()['message'])