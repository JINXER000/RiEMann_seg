import sys
# sys.path.append(".")
sys.path.append("/home/xuhang/Desktop/yzchen_ws/RiEMann_seg/")
import os
import torch
from networks import *
from omegaconf import OmegaConf
from utils.data_utils import downsample_table
import argparse
import numpy as np
from utils.utils import modified_gram_schmidt
import open3d as o3d
import colorsys

class riemann_graper:
    def __init__(self, base_path) -> None:
        self.base_path = base_path
        cfg_path = os.path.join("config", base_path, "config.json")
        all_cfg = OmegaConf.load(cfg_path)
        self.cfg_mani = all_cfg.mani

        model_path = os.path.join("experiments", base_path, "maninet.pth")
        self.policy_mani = globals()[self.cfg_mani.model](voxel_size=self.cfg_mani.voxel_size, \
                                                     radius_threshold=self.cfg_mani.radius_threshold).float().to(self.cfg_mani.device)
        self.policy_mani.load_state_dict(torch.load(model_path))
        self.policy_mani.eval()

        

    def get_world_grasp(self, cpu_xyz, cpu_rgb, pc_center):

        # demo = np.load(pcd_path)
        ref_point_cpu = pc_center
        # todo: correct the format
        input_xyz = torch.tensor(cpu_xyz).float().unsqueeze(0).to(self.cfg_mani.device)
        input_rgb = torch.tensor(cpu_rgb).float().unsqueeze(0).to(self.cfg_mani.device)
        
        # preprossing input to keep the same as training
        data = {
            "xyz": input_xyz[0],
            "rgb": input_rgb[0],
        }
        # data = downsample_table(data)

        xyz = data["xyz"].unsqueeze(0).to(self.cfg_mani.device)
        rgb = data["rgb"].unsqueeze(0).to(self.cfg_mani.device)

        with torch.no_grad():


            ref_point = torch.tensor(ref_point_cpu).float().unsqueeze(0).to(self.cfg_mani.device)

            output_pos, output_direction = self.policy_mani(
                {"xyz": xyz, "rgb": rgb}, 
                reference_point=ref_point, 
                distance_threshold=self.cfg_mani.distance_threshold,
                save_ori_feature = True,
                draw_pcd=True,
                # pcd_name=args.setting,  # required when draw_pcd == True
            )
            out_dir_schmidt = modified_gram_schmidt(output_direction.reshape(-1, 3).T, to_cuda=True)



        pred_pos = output_pos.detach().cpu().numpy().reshape(3)
        pred_rot = out_dir_schmidt.detach().cpu().numpy()

        return pred_pos, pred_rot
    
    def infer_realworld(self, ply_name, vis=True):
        # the pcd path is fixed
        ply_path = os.path.join("data", self.base_path, ply_name)
        obj_pc = o3d.io.read_point_cloud(ply_path)
        cpu_xyz = np.asarray(obj_pc.points)
        cpu_rgb = np.asarray(obj_pc.colors)
        ref_point_cpu = cpu_xyz.mean(axis=0) 

        pred_pos, pred_rot = self.get_world_grasp(cpu_xyz, cpu_rgb, ref_point_cpu)
        print(f"pred_pos: {pred_pos}")

        if vis:
            self.vis_results(cpu_xyz, cpu_rgb, pred_pos, pred_rot)

        
    def vis_results(self, cpu_xyz, cpu_rgb, pred_pos, pred_rot):
        # do visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cpu_xyz)
        pcd.colors = o3d.utility.Vector3dVector(cpu_rgb)

        pred_trans = np.identity(4)
        pred_trans[:3, :3] = pred_rot
        pred_trans[:3, 3] = pred_pos
        coor_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coor_pred = coor_pred.transform(pred_trans)

        colors = np.asarray(pcd.colors)
        hsv_colors = np.array([colorsys.rgb_to_hsv(*color) for color in colors])
        hsv_colors[:, 1] *= 1.4
        hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)
        adjusted_colors = np.array([colorsys.hsv_to_rgb(*color) for color in hsv_colors])
        pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

        world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([pcd, coor_pred, world_coord])


    def test_training_set(self):
        # load npz file
        pcd_path = os.path.join("data", self.base_path, "riemann_focus_demo.npz")

        # see if one instance in the trainning set is ok
        for select_id in range(10):
            demo = np.load(pcd_path)
            cpu_xyz = demo['xyz'][select_id]
            cpu_rgb = demo['rgb'][select_id]

            ref_point_cpu = cpu_xyz.mean(axis=0)  #  Note: cpu_xyz.shape = (1024, 3)
            pred_pos, pred_rot = self.get_world_grasp(cpu_xyz, cpu_rgb, ref_point_cpu)
            print(f"pred_pos: {pred_pos}")

            self.vis_results(cpu_xyz, cpu_rgb, pred_pos, pred_rot)



if __name__ == "__main__":
    base_path = "mug/pick/"
    grasp_gen = riemann_graper(base_path)
    grasp_gen.infer_realworld("graspobj_1.ply")
    # grasp_gen.test_training_set()



