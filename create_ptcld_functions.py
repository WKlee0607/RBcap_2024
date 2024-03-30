import numpy as np
import open3d as o3d

class RGBD:
    def __init__(self, data_type = 0, rgb_path = '', depth_path = '', np_rgb = None, np_depth = None): # basic format
        self.data_type = data_type
        if data_type == 0:
            self.color_path = rgb_path
            self.depth_path = depth_path
        else:
            self.rgb = np_rgb # (C,W,H) format. np.ndarray
            self.depth = np_depth
        
    # rgbd path to o3d.geometry.RGBDImage
    def rgbdpath_to_o3drgbd(self):
        # get rgbd_img as o3d.geometry.RGBDImage Class
        try:
            color_raw = o3d.io.read_image(self.color_path) # color_path: *.jpg or *.png
            depth_raw = o3d.io.read_image(self.depth_path) # depth_path: *.jpg or *.png
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw) #convert_rgb_to_intensity = False)
            self.rgbd_image = rgbd_image
            return self.rgbd_image # o3d.geometry.RGBDImage
        except Exception as e:
            print("Can't convert RGBD to PCD. Error: ", e)
            return None
    
    def nparr_to_o3drgbd(self):
        try:
            o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(np.transpose(self.rgb, (2, 1, 0)))) # (H, W, C)
            o3d_depth = o3d.geometry.Image(np.ascontiguousarray(self.depth.T).astype(np.float32)) # (H, W)
            self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth) #convert_rgb_to_intensity= False)
        except Exception as e:
            print("Can't convert numpy to PCD. Error: ", e)
        return self.rgbd_image
    
    # run
    def run(self): # convert RGB-D to o3d.RGBDImage 
        if self.data_type == 0:  
            return self.rgbdpath_to_o3drgbd() # input rgbd: file path
        else: 
            return self.nparr_to_o3drgbd() # input rgbd: np_array


class PCD:
    def __init__(self,rgbd_image:o3d.geometry.RGBDImage):
        self.rgbd_image = rgbd_image
        self.pcd = None
    
    def create_pcd_from_rgbd(self, intrinsic:o3d.camera.PinholeCameraIntrinsic, extrinsic:np.ndarray): 
        # convert rgbd to point cloud
        

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            self.rgbd_image,
            intrinsic,
            extrinsic
        )
        print(pcd)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.pcd = pcd
        return pcd # o3d.cpu.pybind.geometry.PointCloud
    
     # visualize point cloud
    def visualize_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])

    def run(
            self, 
            intrinsic:o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            extrinsic:np.ndarray = np.eye(4, dtype=np.float64)
            ): # convert o3d.RGBDImage to o3d.Pcd & visualization
        self.create_pcd_from_rgbd(intrinsic, extrinsic)
        self.visualize_pcd()
    




