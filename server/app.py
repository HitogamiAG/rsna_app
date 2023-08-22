import dotenv
import boto3
import base64
from io import BytesIO
from datetime import datetime
import requests
import json

import uuid
from fastapi import FastAPI, Request, Form

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from rds import ServiceHistory, update_value, send_results_to_db, update_spacing

config = dotenv.dotenv_values('.env')


boto_session = boto3.Session(
    region_name='us-east-1',
    aws_access_key_id=config['AWS_SERVER_PUBLIC_KEY'],
    aws_secret_access_key=config['AWS_SERVER_SECRET_KEY'],
)
s3 = boto_session.resource('s3')

engine = create_engine(f"postgresql+psycopg2://{config['AWS_RDS_USERNAME']}:{config['AWS_RDS_PASSWORD']}@{config['AWS_RDS_ENDPOINT']}:{config['AWS_RDS_PORT']}/{config['AWS_RDS_DB_NAME']}")
engine.connect()
session = Session(engine)

app = FastAPI()

@app.get('/')
async def main():
    return {'message': 'Hello from RSNA back'}

@app.post('/upload')
async def upload_ct(filename : str = Form(...), filedata : str = Form(...)):
    file_extension = filename.split('.')[-1]
    
    if file_extension not in ['zip', 'rar', '7z']:
        return {'message': 'ERROR: Incorrect file extension! Supported types: zip, rar, 7z'}
    
    bytes = str.encode(filedata)
    fileid = uuid.uuid4().__str__()
    try:
        bytes_data = base64.b64decode(bytes)
        tempfile = BytesIO()
        tempfile.write(bytes_data)
        tempfile.seek(0)
        
        s3.meta.client.upload_fileobj(tempfile, 'agavrilko-rsna-bucket', fileid + '.' + file_extension)
    except Exception as e:
        print(e)
        return {'message' : 'We faced with error when processing file'}
    
    session.add(ServiceHistory(
        id = fileid,
        filename = filename,
        datetime = datetime.utcnow(),
        state = 'Segmentation: In queue'
    ))
    session.commit()
    requests.post(config['SEGMENTATION_SERVER'], {'filename' : fileid + '.' + file_extension})
    return {'message' : 'Successfully loaded!'}

@app.post('/write_dcm_params')
def write_dcm_params(fileid: str = Form('...'), pixel_spacing_w: str = Form('...'), pixel_spacing_h: str = Form('...'), slice_thickness: str = Form('...')):
    update_spacing(session, fileid,
                   (float(pixel_spacing_w), float(pixel_spacing_h), float(slice_thickness)))

@app.post('/update_status')
def update_status(id : str = Form('...'), value : str = Form('...')):
    update_value(session, id, value)

@app.post('/segm_finish_signal')
def segm_finished(fileid: str = Form('...'), filename: str = Form('...  ')):
    update_status(fileid, 'Segmentation: Done')
    requests.post(config['CLASSIFICATION_SERVER'], {'fileid' : fileid, 'filename' : filename})
    
@app.post('/clf_finish_signal')
def clf_finished(fileid: str = Form('...'), results: str = Form('...')):
    results = json.loads(results)
    update_status(fileid, 'Done')
    send_results_to_db(session, fileid, results)

@app.get('/history')
def get_history():
    results = session.query(ServiceHistory).all()
    return_form = {'message' : []}
    for result in results:
        return_form['message'].append(result)
    return return_form
    
@app.get('/ready_to_visualize')
def ready_to_visualize():
    results = session.query(ServiceHistory).filter(ServiceHistory.state == 'Done').all()
    return_form = {'message' : [result for result in results]}
    return return_form

@app.get('/get_npz')
def get_npz(fileid: str = Form('...')):
    npzfile = BytesIO()
    s3.meta.client.download_fileobj(config['AWS_BUCKET'], fileid + '.npz', npzfile)
    npzfile.seek(0)
    
    pixel_spacing_w, pixel_spacing_h, slice_thickness = session.query(ServiceHistory.pixel_spacing_w,
                        ServiceHistory.pixel_spacing_h,
                        ServiceHistory.slice_thickness).filter(ServiceHistory.id == fileid).one()
    
    return {'npzfile': base64.b64encode(npzfile.getbuffer()),
            'pixel_spacing_w' : str(pixel_spacing_w),
            'pixel_spacing_h' : str(pixel_spacing_h),
            'slice_thickness' : str(slice_thickness)}