from create_ptcld_functions import *
from settings import *
from fn import *
from voxel import *

def main(step, img_dataType):
    datas, intrinsic = get_datas_multiple_timestep(step, img_dataType)
    voxels = Recon3Dpcd(datas, intrinsic)
    voxels.visualize()


    return 



step = 40 # 가져올 이미지 및 센서 데이터 수
img_dataType = 1 # data_type: {0: 기본, 1: TUM, 2:np 입력}
main(step, img_dataType)
