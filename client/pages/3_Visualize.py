import requests
from scipy.sparse import load_npz
import dotenv
import streamlit as st
import io
import base64
import numpy as np
import pyvista as pv
import trimesh

from pyvista.plotting.utilities import xvfb
xvfb.start_xvfb()

import warnings
warnings.filterwarnings('ignore')

from stpyvista import stpyvista

config = dotenv.dotenv_values('.env')

resp = requests.get(config['SERVER_HOST'] + '/ready_to_visualize')
id_list = [result['id'] for result in resp.json()['message']]

option = st.selectbox('Choose record:', id_list)
button = st.button('Visualize')

if option is not None and button:
    status = st.text('Visualization in progress...')
    
    npz_resp = requests.get(config['SERVER_HOST'] + '/get_npz', data={'fileid' : str(option).strip()}).json()
    npzfile = base64.b64decode(npz_resp['npzfile'])
    npzfile = io.BytesIO(npzfile)
    npzfile.seek(0)
    
    pixel_spacing_w, pixel_spacing_h = float(npz_resp['pixel_spacing_w']), float(npz_resp['pixel_spacing_h'])
    slice_thickness = float(npz_resp['slice_thickness'])
    
    print(pixel_spacing_h, pixel_spacing_w, slice_thickness)
    
    matrix = load_npz(npzfile)
    mask_data = matrix.toarray().reshape(512, 512, matrix.shape[-1])
    
    spines = np.unique(mask_data)

    spines_cords = {}

    for spine in spines[1:]:
        indices = np.where(mask_data == spine)
        first_z = indices[2].min()
        last_z = indices[2].max()
        
        spines_cords[spine] = {
            'FirstZ' : first_z,
            'LastZ'  : last_z,
            'ShiftZ' : (first_z * slice_thickness) - (slice_thickness / 2)
        }
    print(spines_cords)
    plotter = pv.Plotter()

    for index, spine in enumerate(spines_cords.keys()):
        print(spine)
        spine_data = mask_data[:, :, spines_cords[spine]['FirstZ']:spines_cords[spine]['LastZ']].copy()
        spine_data[spine_data != spine] = 0
            
        spine_mesh = pv.wrap(trimesh.voxel.VoxelGrid(spine_data).marching_cubes)
        spine_mesh.scale([pixel_spacing_h, pixel_spacing_w, slice_thickness], inplace=True)
        spine_mesh.translate([0, 0, spines_cords[spine]['ShiftZ']], inplace=True)
        spine_mesh.smooth(50, relaxation_factor=0.2, edge_angle=45, inplace=True)
        
        if index % 2 == 0:
            color = [240, 240, 240]
        else:
            color = [0, 127, 0]
        
        plotter.add_mesh(spine_mesh,
                        color= color,#np.random.randint(0, 256, size=3).tolist(),
                        name=f'Spine {spine}')
    
    status.text('Complete')
    
    plotter.background_color = "black"
    plotter.camera.position = (1.1, 1.5, 312.0)
    plotter.camera.focal_point = (0.2, 0.3, 0.3)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.zoom(1.4)
    stpyvista(plotter, key='Spines')
