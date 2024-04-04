# settings 
NYU_LABLED_PATH = "./datas/nyu_depth_v2_labeled.mat"

RGBD_DATASET_ROOT_PATH = "./datas/rgbd_dataset_freiburg2_pioneer_360/"
# 순서대로 읽어오면 됨. 각각이 순서대로 매칭됨.
DEPTH_TXT = RGBD_DATASET_ROOT_PATH + "depth.txt"
RGB_TXT = RGBD_DATASET_ROOT_PATH + "rgb.txt"
GT_TXT = RGBD_DATASET_ROOT_PATH + "groundtruth.txt" # pose info

#TUM Dataset camera params
fx = 520.9  # focal length x
fy = 521.0  # focal length y
cx = 325.1  # optical center x
cy = 249.7  # optical center y
img_height = 480
img_width = 640
