import numpy as np
import open3d as o3d
import matplotlib.image as mpimg
import re


# This is special function used for reading NYU pgm format
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


class RGBD:
    def __init__(self, rgb_path, depth_path): # basic format
        self.color_path = rgb_path
        self.depth_path = depth_path
        if rgb_path[-3:] == "ppm":
            self.img_type = 1
        else:
            self.img_type = 0
    
    def __init__(self, rgbd): # NYU or TUM format
        self.color_path = rgbd.color_path
        self.depth_path = rgbd.depth_path
        if rgbd.color_path[-3:] == "ppm":
            self.img_type = 1
        else:
            self.img_type = 0

    def __init__(self, np_rgb, np_depth): # (C,W,H) format
        self.rgb = np_rgb
        self.depth = np_depth
    
    # rgbd path to o3d.geometry.RGBDImage
    def rgbdpath_to_o3drgbd(self):
        # get rgbd_img as o3d.geometry.RGBDImage Class
        try:
            if self.img_type == 1: # NYU format
                color_raw = mpimg.imread(self.color_path) # *.ppm
                depth_raw = read_nyu_pgm(self.depth_path) # *.pgm
                color = o3d.geometry.Image(color_raw) 
                depth = o3d.geometry.Image(depth_raw) 
                rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
            else:
                color_raw = o3d.io.read_image(self.color_path) # color_path: *.jpg or *.png
                depth_raw = o3d.io.read_image(self.depth_path) # color_path: *.jpg or *.png
                rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
            self.rgbd_image = rgbd_image
            return rgbd_image # o3d.geometry.RGBDImage
        except:
            return
    
    def nparr_to_o3drgbd(self):
        o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(np.transpose(self.rgb, (2, 1, 0)))) # (H, W, C)
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(self.depth.T).astype(np.float32)) # (H, W)
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, convert_rgb_to_intensity= False)
        return self.rgbd_image
    
    def create_pcd_from_rgbd(
        self, 
        intrinsic:o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), 
        extrinsic:np.ndarray = np.eye(4, dtype=np.float64)): 
        # convert rgbd to point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            self.rgbd_image,
            intrinsic,
            extrinsic
        )
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.pcd = pcd
        return pcd # o3d.cpu.pybind.geometry.PointCloud

    # visualize point cloud
    def visualize_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])

    # run
    def run(self, input_type=0):
        if input_type == 0: # rgbd np_array 
            self.nparr_to_o3drgbd()
        else: # rgbd file path
            self.rgbdpath_to_o3drgbd()
        self.create_pcd_from_rgbd()
        self.visualize_pcd()





