from BA2 import *
from fn import *

# BA2 버전
def main(step, img_dataType):
    datas = get_datas_multiple_timestep(step, img_dataType)
    Time_step_imgs = []
    for rgbd_path, depth_path, quaternion, translation in datas:
        step_img = TimeImg(rgbd_path, depth_path, [*quaternion, *translation])
        Time_step_imgs.append(step_img)

    ba = BA(Time_step_imgs)
    res = ba.execute() # BA 실현 -> 35초 정도 걸림 

    return 1


step = 10 # 가져올 이미지 및 센서 데이터 수
img_dataType = 1 # data_type: {0: 기본, 1: TUM, 2:np 입력}
main(step, img_dataType)