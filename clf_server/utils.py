import numpy as np
import cv2
import env as config
import io
import requests
import cv2 as cv
import pydicom as dicom
import json

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
    
def send_finish_signal(fileid: str, results: dict):
    requests.post(config.SERVER_HOST + '/clf_finish_signal', {'fileid' : fileid, 'results' : json.dumps(results)})
    
def update_status(id, value):
    requests.post(config.SERVER_HOST + '/update_status',
                  data={
                      'id' : id,
                      'value' : value
                  })
    
def extract_proportional_elements(arr, T_size = 32):
    length_of_array = len(arr)

    if length_of_array == T_size:
        return arr
    elif length_of_array > T_size:
        step_size = length_of_array / T_size
        current_index = 0
        result_list = []
        for i in range(T_size):
            next_index = int(current_index + step_size)
            if next_index >= length_of_array:
                next_index = length_of_array - 1
            result_list.append(arr[next_index])
            current_index = next_index
        return result_list
    else:
        result_list = []
        num_copies = T_size // length_of_array
        for i in range(num_copies):
            result_list.extend(arr)

        remainder = T_size - len(result_list)

        if remainder != 0:
            step_size = length_of_array / remainder
            current_index = 0
            for i in range(remainder):
                next_index = current_index + step_size
                if next_index >= length_of_array:
                    next_index = length_of_array - 1
                result_list.append(arr[int(next_index)])
                current_index = next_index
        return sorted(result_list)

def compute_bounding_rect(mask, T_size = 32):
    # compute rects
    bRects = [cv.boundingRect(mask[:, :, i]) for i in range(T_size)]
    # compute centers
    bRects = [(rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)) for rect in bRects]
    # compute mean
    x, y = np.array(bRects).mean(axis=0).astype(int)
    # calculate compensation
    x_compensation = max(-(x - 112), 0) + min(-(x + 112 - 512), 0)
    y_compensation = max(-(y - 112), 0) + min(-(y + 112 - 512), 0)
    # apply compensation and use 
    x, y = x + x_compensation, y + y_compensation
    x1, y1, x2, y2 = (
        x - 112, y - 112, x + 112, y + 112
    )
    return (x1, y1, x2, y2)
    
def get_cr_params(mask):
    vrt_dict = {k : [] for k in range(1, 8)}
    for index in range(mask.shape[-1]):
        vrts = np.unique(mask[:, :, index])[1:]
        for vrt in vrts:
            if vrt < 8:
                vrt_dict[vrt].append(index)
            
    for i in range(1, 8):
        if len(vrt_dict[i]) < 6:
            vrt_dict[i] = None
    
    for k, v in vrt_dict.items():
        if v is not None:
            vrt_dict[k] = extract_proportional_elements(v, config.SEQUENCE_SIZE)
    
    to_return = []
    for k, v in vrt_dict.items():
        if v is not None:
            submask = mask[:, :, v]
            submask = (submask == k).astype(np.uint8)
            x, y, _, _ = compute_bounding_rect(submask, config.SEQUENCE_SIZE)
            to_return.append((k, x, y, v))
        else:
            to_return.append((k, None, None, None))
            
    return to_return

def load_dicom(dcmfile: io.BytesIO, coords):
    x, y = coords
    
    dcmfile = dicom.dcmread(dcmfile)
    dcmfile.PhotometricInterpretation = 'YBR_FULL'
    dcmfile = dcmfile.pixel_array
    
    dcmfile = dcmfile[x:x+config.IMG_SIZE, y:y+config.IMG_SIZE]
    
    dcmfile = dcmfile - np.min(dcmfile)
    if np.max(dcmfile) != 0:
        dcmfile = dcmfile / np.max(dcmfile)
    dcmfile=(dcmfile * 255).astype(np.uint8)
    dcmfile = cv.cvtColor(dcmfile, cv.COLOR_GRAY2RGB)
    dcmfile = dcmfile / 255.
    return np.transpose(dcmfile, axes=(2, 0, 1))

def load_dcm_array(dcmzip, dcm_filenames, slices, coords):
    dcm_array = np.zeros((config.BATCH_SIZE, config.SEQUENCE_SIZE, config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
    for index, slice_ in enumerate(slices):
        dcmfile = io.BytesIO(dcmzip.read(dcm_filenames[slice_]))
        dcm_array[:, index] = load_dicom(dcmfile, coords)
    return dcm_array