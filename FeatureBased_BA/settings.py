# settings 
RGBD_DATASET_ROOT_PATH = "datas\\rgbd_dataset_freiburg2_pioneer_360\\"
# 순서대로 읽어오면 됨. 각각이 순서대로 매칭됨.
DEPTH_TXT = RGBD_DATASET_ROOT_PATH + "depth.txt"
RGB_TXT = RGBD_DATASET_ROOT_PATH + "rgb.txt"
GT_TXT = RGBD_DATASET_ROOT_PATH + "groundtruth.txt" # pose info

# device
device = 'CPU:0'

# img size
img_height = 480
img_width = 640

#TUM Dataset camera intrinsic params
fx = 520.9  # focal length x
fy = 521.0  # focal length y
cx = 325.1  # optical center x
cy = 249.7  # optical center y

factor = 5000 # for the 16-bit PNG files
# OR: factor = 1 # for the 32-bit float images in the ROS bag files
'''
for v in range(depth_image.height):
  for u in range(depth_image.width):
    Z = depth_image[v,u] / factor;
    X = (u - cx) * Z / fx;
    Y = (v - cy) * Z / fy;
'''