from fn import *

# 참고: https://www.open3d.org/docs/latest/tutorial/Advanced/rgbd_integration.html

class Recon3DMesh:
    def __init__(self, datas, intrinsic):
        self.datas = datas
        self.intrinsic = intrinsic

        # run 이후 발생
        self.mesh = None
    

    def make_new_volume(self, voxel_length, sdf_trunc, color_type):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length,
            sdf_trunc,
            color_type
        )
        return volume
    
    def volume_integrate(self, volume):
        for idx in range(len(self.datas)):
            rgbd = self.datas[idx][0]
            volume.integrate(
                rgbd,
                intrinsic = self.intrinsic,
                extrinsic = get_ratation_mat(self.datas[idx][1], self.datas[idx][2])
            )
        return volume

    def make_mesh(self, intergrated_volume):
        mesh = intergrated_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def mesh_visualize(self):
        o3d.visualization.draw_geometries([self.mesh])

    def run_mesh(self, voxel_length = 4.0/512.0, sdf_trunc=0.04, color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8):
        volume = self.make_new_volume(voxel_length, sdf_trunc, color_type)
        intergrated_volume = self.volume_integrate(volume)
        self.mesh = self.make_mesh(intergrated_volume)
        return self



# 참조: https://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html#Make-a-combined-point-cloud
class Recon3Dpcd:
    def __init__(self, datas, intrinsic):
        # 초기 멤버 변수
        self.pcd_combined = self.make_recon3Dpcd(datas) # 여러 장의 3D pcds를 combined한 o3d.geometry.PointCloud 객체
        self.pcd_coordinates = self.make_pcd_coordinates() # 3D pcds의 좌표를 저장한 np.array 행렬. shape: (n_3Dpcds, 3)

        # 넣을지 말지 고민중
        self.datas = datas
        self.intrinsic = intrinsic
    
    def make_recon3Dpcd(self, datas): # 무식하게 다 합쳐놓은거. 
        current_rot = datas[0][1]
        current_trans = datas[0][2]
        pcd_combined = o3d.geometry.PointCloud()
        for i in range(len(datas)):
            pcd = PCD(datas[i][0]).run() 
            if i != 0:
                pcd.transform(get_rot_mat_from_q1q2(datas[i][1], current_rot, datas[i][2], current_trans))
            pcd_combined += pcd
        pcd_combined.voxel_down_sample(voxel_size=0.02)
        return pcd_combined
    
    def make_pcd_coordinates(self):
        return np.array(self.pcd_combined.points) 

    def operate_BA(self):
        # 이제 넣어야 함.
        return


    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd_combined])
