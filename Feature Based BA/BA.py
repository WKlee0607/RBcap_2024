import cv2
import open3d as o3d
import numpy as np

from settings import *



class TimeImg:
    def __init__(self, img, dep_img, init_poses): 
        '''
        - img: rgb img file path
        - dep_img: depth img file path
        - init_poses: world coord RT mat. <- 3x4 행렬
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
    def __init__(self, input_TimeImg):
        '''
        1. input_TimeImg: list [t-T-1, t-T, t-T+1, ..., t] -> T개
        2. good_matched_features: list [[goodmatches(t-T-1,t-T)], ... ,[goodmatches(t-1,t)]] -> T-1개
        3. init_RT_mat: list [RT(t-T-1, t-T), ... , RT(t-1, t)] -> T-1개
        4. init_3d_pcds: list [np.array[N1,3], np.array[N2,3], ..., np.array[Nt,3]] -> T-1개
        5. measured_2d_pts: list [[pts1, pts2], ...., [pts1, pts2]] -> T-1개, 원소는 크기가 2인 list <- matched pts 좌표임
        6. proj_2d_pts: list [[pts1, pts2], ...., [pts1, pts2]] -> T-1개, 원소는 크기가 2인 list
        '''
        self.input_TimeImg = input_TimeImg
        self.good_matched_features = self.feature_matching()

        #self.init_RT_mat = self.get_init_RT_using_kp()
        self.init_3d_pcds = self.get_init_3d_pcds()
        self.measured_2d_pts = self.get_measured_2d_coords()
        self.proj_2d_pts = self.get_proj_pts()

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
    
    def get_measured_2d_coords(self):
        # good matched pts coords 저장
        measured_2d_coors = []
        for i in range(len(self.good_matched_features)):
            m_list = self.good_matched_features[i]
            prev_pts = self.input_TimeImg[i].kp
            curr_pts = self.input_TimeImg[i+1].kp

            pts1 = np.float32([prev_pts[m.queryIdx].pt for m in m_list]).reshape(-1, 2) # (N, 2)
            pts2 = np.float32([curr_pts[m.trainIdx].pt for m in m_list]).reshape(-1, 2)

            measured_2d_coors.append([pts1, pts2])
        return measured_2d_coors

    def get_init_RT_using_kp(self):
        '''
        이전에서 다음 스텝으로 가는 init Rt구하기 <- matched key pts이용해서 구함
        -> 혹시나 odometry 이용 못할 시 사용하려고 만든거긴함.
        '''
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix
        RTs = []

        for i in range(len(self.good_matched_features)):
            m_list = self.good_matched_features[i]
            prev_pts = self.input_TimeImg[i].kp
            curr_pts = self.input_TimeImg[i+1].kp

            pts1 = np.float32([prev_pts[m.queryIdx].pt for m in m_list]).reshape(-1, 1, 2)
            pts2 = np.float32([curr_pts[m.trainIdx].pt for m in m_list]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask) # -> matched kps로 R,t 구한것
            RT = np.hstack((R, t))
            RTs.append(RT)
        return RTs


    def get_init_3d_pcds(self):
        '''
        - Triangulate 이용.
        - 이 때 init pose는 odometry값(World Coord) 그대로 이용.
        '''
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix
        pcds = []

        for i in range(len(self.input_TimeImg)-1):
            prev_proj = K @ self.input_TimeImg[i].init_poses # 3d pose(world coor)
            curr_proj = K @ self.input_TimeImg[i+1].init_poses

            m_list = self.good_matched_features[i]
            prev_pts = self.input_TimeImg[i].kp
            curr_pts = self.input_TimeImg[i+1].kp

            pts1 = np.float32([prev_pts[m.queryIdx].pt for m in m_list]).reshape(-1, 1, 2)
            pts2 = np.float32([curr_pts[m.trainIdx].pt for m in m_list]).reshape(-1, 1, 2)

            points_4D = cv2.triangulatePoints(prev_proj, curr_proj, pts1, pts2)
            points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
            points_3D = points_3D[:3, :].T # (N, 3)
            pcds.append(points_3D)
        return pcds

    def visualization(self):
        combined_pcds = o3d.geometry.PointCloud()
        for i in range(len(self.init_3d_pcds)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.init_3d_pcds[i])
            combined_pcds+=pcd
        print(combined_pcds)
        o3d.visualization.draw_geometries([combined_pcds])
        
    def get_proj_pts(self):
        # projection -> prev, curr 둘 다 reproject해주기
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix

        proj_2d_coords = []
        for i in range(len(self.init_3d_pcds)):
        # project 3d pcd
            prev_proj_mat = K @ self.input_TimeImg[i].init_poses # (3, 4)
            curr_proj_mat = K @ self.input_TimeImg[i+1].init_poses
            
            pcd = np.hstack((self.init_3d_pcds[i], np.ones((len(self.init_3d_pcds[i]),1)))).T # (4, N)

            prev_proj_pts = prev_proj_mat @ pcd # (3, N)
            curr_proj_pts = curr_proj_mat @ pcd
            
            prev_proj_pts = (prev_proj_pts / prev_proj_pts[2])[:2].T # (N, 2)
            curr_proj_pts = (curr_proj_pts / curr_proj_pts[2])[:2].T
            proj_2d_coords.append([prev_proj_pts, curr_proj_pts])

        return proj_2d_coords

    def compute_residual(self, ):
        return






'''
# input: n장 이미지, n장 init pose
class BA:
    def __init__(self, input_imgs, init_poses):
        
        #input_imgs: list 
        #init_poses: list
        
        self.input_imgs = input_imgs
        self.init_poses = init_poses

    def feature_ext(self):
        
        #1. cvt gray
        #2. get feature ext
        
        kps = []
        dess = []
        orb = cv2.ORB_create()
        for img in self.input_imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, des = orb.detectAndComput(gray, None)
            kps.append(kp)
            dess.append(des)
        
        self.kps = kps
        self.dess = dess
        return kps, dess
    
    def feature_matching(self):
        bf = cv2.BFMatcher()
        dist_coef = 0.8
        self.good_matched_features = []

        for i in range(len(self.kps)-1):
            matches = bf.knnMatch(self.dess[i], self.dess[i+1], k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < dist_coef * n.distance:
                    good_matches.append(m)
            self.good_matched_features.append(good_matches)
        return self.good_matched_features
    
    def triangulation_from_matched_features(self):
        # ref: https://www.opencvhelp.org/tutorials/advanced/reconstruction-opencv/
        


        return
    
'''