import h5py

from create_ptcld_functions import *
from settings import *

f = h5py.File(NYU_LABLED_PATH)
#print(f.keys()) # ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']>
#print(f['images']) # (1449, 3, 640, 480)
#print(f["depths"].shape) # (1449, 640, 480)

idx = 5
rgb_raw = f['images'][idx] # C, W, H
depth_raw = f['depths'][idx]

rgbd=RGBD(rgb_raw, depth_raw)
rgbd.run() # 3D representation

