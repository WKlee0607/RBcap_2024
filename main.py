import cv2

from create_ptcld_functions import *
from settings import *
from fn import *


rgb_path_list = open_txt_file(RGB_TXT) # len: 1225
depth_path_list = open_txt_file(DEPTH_TXT) # len: 1209
gt_path_list = open_txt_file(GT_TXT) # len: 21823

idx = 0
rgb_path = get_path(idx ,rgb_path_list, "rgb")
depth_path = get_path(idx, depth_path_list, "")

intrinsic = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
pcd = PCD(RGBD(0, rgb_path, depth_path).run()).run()
