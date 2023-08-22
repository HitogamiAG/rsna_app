import boto3
import io
import gc
import zipfile
import time
import queue
import requests

import numpy as np
import pydicom as dicom
import onnxruntime as ort

from threading import Thread

from fastapi import FastAPI, Form

from utils import (image_preprocessing,
                   convert_to_sparse,
                   save_to_s3,
                   update_status,
                   send_finish_signal)
import env as config

session = boto3.Session(
    region_name='us-east-1',
    aws_access_key_id=config.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=config.AWS_SERVER_SECRET_KEY
)
s3 = session.resource('s3')

segmentation_queue = queue.Queue()

app = FastAPI()

@app.post('/')
def add_file_to_queue(filename: str = Form(...)):
    segmentation_queue.put(filename)
    return {}

sess = ort.InferenceSession(
    config.MODEL_PATH,
    providers=ort.get_available_providers()
)
input_name = sess.get_inputs()[0].name

def main(segmentation_queue):
    while True:
        if not segmentation_queue.empty():
            filename = segmentation_queue.get()
            fileid = '.'.join(filename.split('.')[:-1])
            
            tempfile = io.BytesIO()
            s3.meta.client.download_fileobj(config.AWS_BUCKET, filename, tempfile)
            
            update_status(fileid, 'Segmentation: In work')
            tempfile = zipfile.ZipFile(tempfile)
            filenames = tempfile.namelist()
            filenames = [file for file in filenames if '.dcm' in file]
            filenames = sorted(filenames, key = lambda x: int(x.split('/')[-1][:-4]))
            num_files = len(filenames)
            
            dcmfile = io.BytesIO(tempfile.read(filenames[0]))
            dcmfile = dicom.dcmread(dcmfile)
            print(dcmfile.PixelSpacing[0], dcmfile.PixelSpacing[1], dcmfile.SliceThickness)
            print(type(dcmfile.PixelSpacing[0]), type(dcmfile.SliceThickness))
            requests.post(config.SERVER_HOST + '/write_dcm_params', {'fileid' : fileid,
                'pixel_spacing_w' : str(dcmfile.PixelSpacing[0]),
                'pixel_spacing_h' : str(dcmfile.PixelSpacing[1]),
                'slice_thickness' : str(dcmfile.SliceThickness)})

            filenames = [filenames[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE] \
                for i in range((len(filenames) + config.BATCH_SIZE - 1) // config.BATCH_SIZE )]

            mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE, num_files))

            for batch_idx, chunk in enumerate(filenames):
                img = np.zeros((len(chunk), config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
                
                for index, dcm_filename in enumerate(chunk):
                    dcmfile = io.BytesIO(tempfile.read(dcm_filename))
                    dcmfile = dicom.dcmread(dcmfile)
                    dcmfile.PhotometricInterpretation = 'YBR_FULL'
                    img[index] = image_preprocessing(dcmfile.pixel_array)
                
                pred_onx = sess.run(None, {input_name : img.astype(np.float32)})[0]
                pred_onx = np.transpose(pred_onx.argmax(axis = 1), axes = (1, 2, 0))
                
                mask[:, :, config.BATCH_SIZE * batch_idx : config.BATCH_SIZE * batch_idx + len(chunk)] = \
                    pred_onx
                    
                update_status(fileid, f'Segmentation: {batch_idx + 1}/{len(filenames)}')
                
            sparse_matrix = convert_to_sparse(mask)
            
            save_to_s3(sparse_matrix, fileid, s3)
            send_finish_signal(fileid, filename)
            
            del tempfile, sparse_matrix, img, mask
            gc.collect()
        else:
            time.sleep(1)

worker = Thread(target=main, args=[segmentation_queue], daemon=True)
worker.start()

# with open('test_data/dcm.txt', 'rb') as outfile:
#     tempfile = io.BytesIO(outfile.read())

# tempfile = zipfile.ZipFile(tempfile)
# filenames = tempfile.namelist()
# filenames = [file for file in filenames if '.dcm' in file]
# filenames = sorted(filenames, key = lambda x: int(x.split('/')[-1][:-4]))
# num_files = len(filenames)
# # Division to batches
# filenames = [filenames[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE] \
#     for i in range((len(filenames) + config.BATCH_SIZE - 1) // config.BATCH_SIZE )]

# mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE, num_files))

# for batch_idx, chunk in enumerate(filenames):
#     img = np.zeros((len(chunk), config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
    
#     for index, filename in enumerate(chunk):
#         dcmfile = io.BytesIO(tempfile.read(filename))
#         dcmfile = dicom.dcmread(dcmfile)
#         dcmfile.PhotometricInterpretation = 'YBR_FULL'
#         img[index] = image_preprocessing(dcmfile.pixel_array)
    
#     pred_onx = sess.run(None, {input_name : img.astype(np.float32)})[0]
#     pred_onx = np.transpose(pred_onx.argmax(axis = 1), axes = (1, 2, 0))
    
#     mask[:, :, config.BATCH_SIZE * batch_idx : config.BATCH_SIZE * batch_idx + len(chunk)] = \
#         pred_onx
        
# sparse_matrix = convert_to_sparse(mask)
# save_to_s3(sparse_matrix, filename[:-4], s3)
