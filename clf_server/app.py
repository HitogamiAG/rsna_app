import boto3
import io
import gc
import zipfile
import time
import queue

import numpy as np
import pydicom as dicom
import onnxruntime as ort

from scipy.sparse import load_npz

from threading import Thread

from fastapi import FastAPI, Form

from utils import (image_preprocessing,
                   update_status,
                   send_finish_signal,
                   get_cr_params,
                   load_dcm_array)
import env as config

session = boto3.Session(
    region_name='us-east-1',
    aws_access_key_id=config.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=config.AWS_SERVER_SECRET_KEY
)
s3 = session.resource('s3')

classification_queue = queue.Queue()

app = FastAPI()

@app.post('/')
def add_file_to_queue(fileid: str = Form('...'), filename: str = Form(...)):
    classification_queue.put((fileid, filename)) # format?

sess = ort.InferenceSession(
    config.MODEL_PATH,
    providers=ort.get_available_providers()
)
input_name = sess.get_inputs()[0].name

def main(classification_queue):
    while True:
        if not classification_queue.empty():
            fileid, filename = classification_queue.get()
            
            update_status(fileid, 'Classification: Load')
            
            dcmzip = io.BytesIO()
            s3.meta.client.download_fileobj(config.AWS_BUCKET, filename, dcmzip)
            dcmzip.seek(0)

            mask_array = io.BytesIO()
            s3.meta.client.download_fileobj(config.AWS_BUCKET, fileid + '.npz', mask_array)
            mask_array.seek(0)  

            update_status(fileid, 'Classification: In work')

            dcmzip = zipfile.ZipFile(dcmzip)
            dcm_filenames = dcmzip.namelist()
            dcm_filenames = [dcm_file for dcm_file in dcm_filenames if '.dcm' in dcm_file]
            dcm_filenames = sorted(dcm_filenames, key = lambda x: int(x.split('/')[-1][:-4]))

            mask_array = load_npz(mask_array)
            mask_array = mask_array.toarray().reshape(config.MASK_IMG_SIZE, config.MASK_IMG_SIZE, -1)

            crop_resize_parameters = get_cr_params(mask_array) # [(vrt, x, y, slices)]

            final_result = {k : None for k in range(1, 8)}
            for params in crop_resize_parameters:
                vrt, x, y, slices = params
                if x is None:
                    continue
                dcm_array = load_dcm_array(dcmzip, dcm_filenames, slices, (x, y))
                
                pred_onx = sess.run(None, {input_name : dcm_array.astype(np.float32)})[0]
                prob = np.squeeze(pred_onx, axis=0).mean().item()
                damaged = int(prob > 0.5)
                
                final_result[vrt] = [damaged, round(prob, 2)]
                
                update_status(fileid, f'Classification: {vrt}/7')
            
            
            send_finish_signal(fileid, final_result)
            
            del dcmzip, mask_array, dcm_array
            gc.collect()
        else:
            time.sleep(1)

worker = Thread(target=main, args=[classification_queue], daemon=True)
worker.start()

# fileid, filename = 'b83d0bf4-c923-475b-a6d3-9e9cc211e4cf', 'b83d0bf4-c923-475b-a6d3-9e9cc211e4cf.zip'

# dcmzip = io.BytesIO()
# # s3.meta.client.download_fileobj(config.AWS_BUCKET, filename, dcmzip)

# with open('test_data/dicom_arr.txt', 'rb') as file:
#     dcmzip.write(file.read())
    
# dcmzip.seek(0)

# mask_array = io.BytesIO()
# # s3.meta.client.download_fileobj(config.AWS_BUCKET, fileid + '.npz', mask_array)

# with open('test_data/mask_array.txt', 'rb') as file:
#     mask_array.write(file.read())

# mask_array.seek(0)  

# # update_status(fileid, 'Classification: In work')

# dcmzip = zipfile.ZipFile(dcmzip)
# dcm_filenames = dcmzip.namelist()
# dcm_filenames = [dcm_file for dcm_file in dcm_filenames if '.dcm' in dcm_file]
# dcm_filenames = sorted(dcm_filenames, key = lambda x: int(x.split('/')[-1][:-4]))
# dcm_num_files = len(dcm_filenames)

# mask_array = load_npz(mask_array)
# mask_array = mask_array.toarray().reshape(config.MASK_IMG_SIZE, config.MASK_IMG_SIZE, -1)

# crop_resize_parameters = get_cr_params(mask_array) # [(vrt, x, y, slices)]

# final_result = {k : None for k in range(1, 8)}
# for params in crop_resize_parameters:
#     vrt, x, y, slices = params
#     if x is None:
#         continue
#     dcm_array = load_dcm_array(dcmzip, dcm_filenames, slices, (x, y))
    
#     pred_onx = sess.run(None, {input_name : dcm_array.astype(np.float32)})[0]
#     prob = np.squeeze(pred_onx, axis=0).mean().item()
#     damaged = int(prob > 0.5)
    
#     final_result[vrt] = (damaged, round(prob, 2))