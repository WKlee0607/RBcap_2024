import cv2
import open3d as o3d
import numpy as np
import quaternion
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from settings import *
from fn import *


class TimeImg:
    def __init__(self, img, dep_img, init_poses): 
        '''
        - img: rgb img file path
        - dep_img: depth img file path
        - init_poses: [w, x, y, z, tx, ty, tz]
        '''
        self.img = np.asarray(o3d.io.read_image(img)) # HWC -> RGB
        self.dep_img = np.asarray(o3d.io.read_image(dep_img))

        self.init_poses = init_poses
        self.kp, self.des = self.feature_ext(self.img)

    def feature_ext(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(gray, None) # 대략 500개 가까이 추출
        return kp, des


class BA:
    # 
    def __init__(self, input_TimeImg):
        '''
        1. input_TimeImg: list [t-T-1, t-T, t-T+1, ..., t] -> 이미지 장수 개
        2. good_matched_features: list [[goodmatches(t-T-1,t-T)], ... ,[goodmatches(t-1,t)]] -> 이미지 장수-1개
        3. init_poses: np.array[이미지 장수, 7] 

        4. indicies: np.array[2d pts개수] -> 몇 번째 3d pts와 연관되는지 idx기입.
        5. pose_indicies: np.array[2d pts개수] -> 이 2d 좌표가 어느 이미지에서 나왔는지 idx기입.
        '''
        self.length = len(input_TimeImg)
        self.input_TimeImg = input_TimeImg
        self.good_matched_features = self.feature_matching()
        
        self.init_poses = self.get_init_poses() # camera_params (N, 7)
        self.init_2d_coords, self.init_3d_coords, self.indicies, self.pose_indicies  = self.get_init_2d_3d_coords()

    def feature_matching(self):
        bf = cv2.BFMatcher()
        dist_coef = 0.5
        good_matched_features = []

        for i in range(len(self.input_TimeImg)-1):
            matches = bf.knnMatch(self.input_TimeImg[i].des, self.input_TimeImg[i+1].des, k=2)
            good_matches = [] # 약 400개씩 뽑힘
            for m, n in matches:
                if m.distance < dist_coef * n.distance:
                    good_matches.append(m)
            good_matched_features.append(good_matches)
        return good_matched_features
    
    def get_init_poses(self):
        res = np.zeros((self.length, 7))
        for i in range(self.length):
            pose = np.array(self.input_TimeImg[i].init_poses)
            res[i] = pose
        return res
    
    def get_init_2d_3d_coords(self):
        init_2d_coors = []
        init_3d_coors = []
        indicies = []
        pose_indicies = []

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix
        cnt_idx = 0

        for i in range(self.length - 1):
            m_list = self.good_matched_features[i]
            prev_pts = self.input_TimeImg[i].kp
            curr_pts = self.input_TimeImg[i+1].kp

            pts1 = np.float32([prev_pts[m.queryIdx].pt for m in m_list]).reshape(-1, 1, 2) # (N,1,2)
            pts2 = np.float32([curr_pts[m.trainIdx].pt for m in m_list]).reshape(-1, 1, 2) # (N,1,2)

            prev_proj = K @ get_ratation_mat(np.quaternion(*self.init_poses[i,:4]), self.init_poses[i,4:])
            curr_proj = K @ get_ratation_mat(np.quaternion(*self.init_poses[i+1,:4]), self.init_poses[i+1,4:])
            
            points_4D = cv2.triangulatePoints(prev_proj, curr_proj, pts1, pts2)
            points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
            points_3D = points_3D[:3, :].T # (N, 3)

            pts1 = np.squeeze(pts1, axis=1)
            pts2 = np.squeeze(pts2, axis=1)

            # 여기서 2d랑 3d랑 좌표 연관시켜줘야함.
            pcd_length = len(points_3D)
            idx = 2*list(range(cnt_idx, cnt_idx+pcd_length))
            cnt_idx += pcd_length
            idxs = np.array(idx) # (N,) 사이즈
            indicies.append(idxs) # list 추가
            
            # 2d 좌표가 어떤 poses를 참조하는지 idx 만들기
            pose_indicies.append(pcd_length*[i,] + pcd_length*[i+1,])
            
            # 2d 좌표 및 3d 좌표 저장
            res = np.vstack((pts1, pts2))
            init_2d_coors.append(res)
            init_3d_coors.append(points_3D)
        
        init_2d_coors = np.vstack(init_2d_coors) # (2d개수, 2)
        init_3d_coors = np.vstack(init_3d_coors) # (3d개수, 3)
        indicies = np.hstack(indicies) # (2d개수) -> 2d 점이 어떤 3d 점과 연관되는지 나타낸 array
        pose_indicies = np.hstack(pose_indicies)

        return init_2d_coors, init_3d_coors, indicies, pose_indicies


    def feature3d_visualization(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.init_3d_coords)
        print(pcd)
        o3d.visualization.draw_geometries([pcd])
        
    def get_proj_3dpts(self, ip_pcd, ip_poses):
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix
        RT = [K @ get_ratation_mat(np.quaternion(*ip_poses[i,:4]), ip_poses[i,4:]) for i in range(len(ip_poses))]

        proj_2d_coords = []
        ip_pcd = np.hstack((ip_pcd, np.ones((len(ip_pcd), 1))))

        for i in range(len(self.indicies)):
            pcd_idx = self.indicies[i]
            pose_idx = self.pose_indicies[i]
            
            rt = RT[pose_idx] # (3,4)
            pcd = np.array([ip_pcd[pcd_idx]]).T # (4,1)

            proj = rt @ pcd # (3,1)
            proj = ((proj / proj[2])[:2].T)[0] # (2,)
            proj_2d_coords.append(proj)

        proj_2d_coords = np.vstack(proj_2d_coords) # (2d개수 , 2)
        return proj_2d_coords
    
    def get_params(self):
        n_cameras = len(self.input_TimeImg)
        n_points = len(self.init_3d_coords)
        camera_indicies = self.pose_indicies
        point_indices = self.indicies
        points_2d = self.init_2d_coords
        return n_cameras, n_points, camera_indicies, point_indices, points_2d

    def compute_residual(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
    
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))
        points_3d = params[n_cameras * 7:].reshape((n_points, 3))
        points_proj = self.get_proj_3dpts(points_3d, camera_params)
        return (points_proj - points_2d).ravel()
    
    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = len(camera_indices) * 2 # residual 계산할 변수 개수 -> (10320)
        n = n_cameras * 7 + n_points * 3 # 조정될 파라미터 개수 -> (7810)
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(len(camera_indices))
        for s in range(7):
            A[2 * i, camera_indices * 7 + s] = 1
            A[2 * i + 1, camera_indices * 7 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

        return A
    
    def execute(self):
        x0 = np.hstack((self.init_poses.ravel(), self.init_3d_coords.ravel())) 
        params = self.get_params()

        f0 = self.compute_residual(x0, *params)
        A = self.bundle_adjustment_sparsity(params[0], params[1], params[2], params[3])

        t0 = time.time()
        res = least_squares(self.compute_residual, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(params[0], params[1], params[2], params[3], params[4]))
        t1 = time.time()

        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        return res






