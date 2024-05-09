import numpy as np
import open3d as o3d
import quaternion

from settings import *

def open_txt_file(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        strings = f.readlines() # list, path끝에는 \n이라 지워줘야함.
        del strings[:3] # 쓸데없는 string들 지워주기.
        return strings.copy() # list 형식


def get_path(idx, path_list:list):
    path = path_list[idx].split()[1]
    split_root = path.split("/")
    last_path = "\\".join(split_root)

    return RGBD_DATASET_ROOT_PATH + last_path

def get_datas_multiple_timestep(step, img_dataType): # TUM Dataset 한정
    rgb_path_list = open_txt_file(RGB_TXT) # len: 1225
    depth_path_list = open_txt_file(DEPTH_TXT) # len: 1209
    gt_path_list = open_txt_file(GT_TXT) # len: 21823

    datas = []
    for idx in range(step):
        line = gt_path_list[idx].split()
        rgbd_path = get_path(idx ,rgb_path_list)
        depth_path = get_path(idx, depth_path_list)
        #quaternion = np.quaternion(float(line[-1]), float(line[-4]), float(line[-3]), float(line[-2])) # np.quaternion(w, x, y, z)
        #translation = np.array([float(line[1]), float(line[2]), float(line[3])]) # np.array([tx, ty, tz])

        # BA2 버전
        quaternion =  [float(line[-1]), float(line[-4]), float(line[-3]), float(line[-2])] # [w, x, y, z]
        translation = [float(line[1]), float(line[2]), float(line[3])] # [tx, ty, tz]
        #


        datas.append([rgbd_path, depth_path, quaternion, translation]) # [o3d.geometry.RGBDImage, np.quaternion, np.array]
    return datas


def get_ratation_mat(input_quaternion:np.quaternion, input_translation:np.array):
    '''
    input rotation(quaternion form)과
    input tralsation matrix를 입력해주면 
    이 둘을 합쳐서 Transform Matrix로 만들어줌. -> 3x4
    '''
    Rot = quaternion.as_rotation_matrix(input_quaternion) # return 회전 행렬: 3X3
    Trans = np.array([input_translation])
    RT = np.hstack([Rot,Trans.T]).astype(np.float64) # 3x4 -> world coords로 이동하는 Mat임.
    return RT

    
def get_rot_mat_from_q1q2(source_quat:np.quaternion, target_quat:np.quaternion, source_trans:np.array, target_trans:np.array):
    '''
    여기서 source는 t시점의 절대 좌표(현재 좌표)
    target은 이전 시점의 절대 좌표(이전 좌표)
    source와 target의 절대 좌표 차이를 이용하여 두 Camera 간의 Transform Matrix를 구함.
    '''
    norm_q1 = source_quat.conjugate()/(source_quat.w**2 + source_quat.x**2 + source_quat.y**2 + source_quat.z**2)
    r = target_quat * norm_q1
    t = target_trans - source_trans
    return get_ratation_mat(r,t)


    