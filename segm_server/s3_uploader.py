import boto3
from scipy.sparse import load_npz, save_npz
import env as config

import io

session = boto3.Session(
    region_name='us-east-1',
    aws_access_key_id=config.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=config.AWS_SERVER_SECRET_KEY
)
s3 = session.resource('s3')

matrix = load_npz('test_data/mask.npz')

tempfile = io.BytesIO()
save_npz(tempfile, matrix)
tempfile.seek(0)

s3.meta.client.upload_fileobj(tempfile, config.AWS_BUCKET, 'test_mask.npz')