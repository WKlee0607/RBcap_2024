from settings import *

def open_txt_file(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        strings = f.readlines() # list, path끝에는 \n이라 지워줘야함.
        del strings[:3] # 쓸데없는 string들 지워주기.
        return strings.copy() # list 형식


def get_path(idx, path_list:list, type:str):
    if type == "rgb":
        return RGBD_DATASET_ROOT_PATH + path_list[idx].split()[1]
    else:
        return RGBD_DATASET_ROOT_PATH + path_list[idx].split()[1]
