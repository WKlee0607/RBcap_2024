import numpy as np
import open3d as o3d
import quaternion

from settings import*
from fn import *


class RGBDPCD:
    def __init__(self, RGBD_PATH, intrinsic, extrinsic):
        '''
        - RGBD_PATH = (RGB_PATH, D_PATH)
        - intrinsic = [fx, fy, cx, cy, k1, k2]
        - extrinsic = [tx ty tz qx qy qz qw]
        '''

        self.RGB = self.get_img_from_path(RGBD_PATH[0])
        self.D = self.get_img_from_path(RGBD_PATH[1]) 

        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

        self.pixel_coord = self.get_2d_coord(self.D) # np.array([height*width, 2]) # u,v
        self.pcd = self.get_3d_pcd(self.intrinsic, self.extrinsic) # np.array([h*w, 3])


    def get_img_from_path(self, PATH):
        return np.asarray(o3d.t.io.read_image(PATH)) # np.ndarray - HWC

    def get_2d_coord(self, img):
        height = img.shape[0] 
        width = img.shape[1] 

        coords = np.zeros((height*width, 2)) # rows 가 더 느리게 내려감
        for i in range(0, height*width, width):
            coords[i:i+width, 0] += int(i/width)
            coords[i:i+width, 1] += range(width)
        return coords # image plane

    
    def get_4x4ratation_mat(self, extrinsic):
        '''
        img 위치 이동시킬 때 사용
        -> 생각해보니 굳이 움직일 필요 없을듯?
        -> 3D point projection할 때, camera extrinsic으로 다시 projection 해주니까?
        '''
        qt = np.quaternion(extrinsic[-1], extrinsic[-4], extrinsic[-3], extrinsic[-2])
        ts = np.array([extrinsic[0], extrinsic[1], extrinsic[2]])

        Rot = quaternion.as_rotation_matrix(qt) # return 회전 행렬: 3X3
        Trans = np.array([ts])
        RT = np.vstack([np.hstack([Rot,Trans.T]), np.array([0,0,0,1])]).astype(np.float64) # size: 4x4
        return RT

    def get_3d_pcd(self, intrinsic, extrinsic):
        h,w = self.D.shape[0:2]
        D = np.array([np.reshape(self.D, (h*w))]).T # (h*w, 1)

        # Set Pixel Coords to Normalized image plane
        fx, fy = self.intrinsic[:2]
        cx, cy = self.intrinsic[2:4]

        img_coor = self.pixel_coord.copy()
        img_coor[:, 0] = (img_coor[:, 0] - cx)/fx
        img_coor[:, 1] = (img_coor[:, 1] - cy)/fy

        # apply radial distortion
        r2 = img_coor[:, 0]**2 + img_coor[:, 1]**2 
        k1, k2 = self.intrinsic[-2:]
        K = np.array([(1 + k1*r2 + k2*(r2**2)).T])
        img_coor = img_coor/K.T

        pcd = img_coor * D
        pcd = np.hstack((pcd, D)) # world 좌표계

        # D가 1보다 작은 것들은 빼야함.
        outlier = np.where(D <= 1)
        pcd = np.delete(pcd, outlier, axis=0)
        self.pixel_coord = np.delete(self.pixel_coord, outlier, axis=0)

        
        pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))

        # apply extrinsic params
        extrinsic_mat = self.get_4x4ratation_mat(self.extrinsic)
        pcd = np.matmul(extrinsic_mat, pcd.T)
        pcd, k = np.split(pcd.T, [3], axis=1)
        pcd = pcd/k
        return pcd
    

    def get_reproj_pixel_coord(self):
        # inv extrinsic
        extrinsic_mat_inv = np.linalg.inv(self.get_4x4ratation_mat(self.extrinsic))
        pcd = np.hstack(pcd, np.ones((pcd.shape[0], 1)))
        pcd = np.matmult(extrinsic_mat_inv, pcd.T)

        pcd, k = np.split(pcd.T, [3], axis=1)
        pcd = pcd/k

        # projection
        xy, z = np.split(pcd, [2], axis=1)
        xy = xy/z

        r2 = xy[:, 0]**2 + xy[:, 1]**2 
        k1, k2 = self.intrinsic[-2:]
        K = np.array([(1 + k1*r2 + k2*(r2**2)).T])


        fx, fy = self.intrinsic[:2]
        cx, cy = self.intrinsic[2:4]
        xy = xy*K.T
        xy[:, 0] = xy[:, 0]*fx + cx
        xy[:, 1] = xy[:, 1]*fy + cy

        return xy # image plane coord - pixel 좌표 -> (rows, 2)


        


