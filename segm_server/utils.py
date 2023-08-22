import numpy as np
import cv2
import env as config
import io
import requests

from scipy.sparse import csr_matrix, save_npz, load_npz

def image_preprocessing(img : np.array) -> np.array:
        """Preprocess image before prediction

        Args:
            img (np.array): Expected numpy array of shape (H, W)

        Returns:
            np.array: Numpy array of shape (C, H, W)
        """
        img[img < 0] = 0
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.transpose(img, axes = (2, 0, 1))
        img = img / 255.
        img = img - 0.5
        img = img / 0.5
        
        return img
    
def convert_to_sparse(mask):
    matrix = csr_matrix(mask.reshape(-1, mask.shape[-1]))
    return matrix

def save_to_s3(matrix, fileid, s3_session):
    tempfile = io.BytesIO()
    save_npz(tempfile, matrix)
    tempfile.seek(0)
    
    s3_session.meta.client.upload_fileobj(tempfile, config.AWS_BUCKET, fileid + '.npz')
    
def send_finish_signal(fileid: str, filename: str):
    requests.post(config.SERVER_HOST + '/segm_finish_signal', {'fileid' : fileid, 'filename' : filename})
    
def update_status(id, value):
    requests.post(config.SERVER_HOST + '/update_status',
                  data={
                      'id' : id,
                      'value' : value
                  })