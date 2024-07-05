# load demo, save all point clouds, visualize seg_center and axes on the pc. 


import open3d as o3d
import numpy as np
import os
import colorsys
import sys
sys.path.append(".")
from utils.data_utils import SE3Demo

def main(demo_path):
    
    demo = SE3Demo(demo_path, data_aug=False, device="cuda:0")

    for i in range(len(demo)):
        gt_trans = np.identity(4)
        gt_trans[:3, :3] = demo[i]['axes'].reshape(-1, 3).cpu().numpy()
        gt_trans[:3, 3] = demo[i]['seg_center'].reshape(3).cpu().numpy()
        xyz = demo[i]['xyz'].cpu().numpy()
        rgb = demo[i]['rgb'].cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        coor_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coor_pred = coor_pred.transform(gt_trans)



    # result = np.load(result_path)
    # pred_pos = result["pred_pos"]
    # pred_rot = result["pred_rot"]
    # pred_trans = np.identity(4)
    # pred_trans[:3, :3] = pred_rot
    # pred_trans[:3, 3] = pred_pos

    # pcd = np.load(pcd_path)
    # xyz = pcd["xyz"]
    # rgb = pcd["rgb"]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    # coor_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # coor_pred = coor_pred.transform(pred_trans)

        colors = np.asarray(pcd.colors)
        hsv_colors = np.array([colorsys.rgb_to_hsv(*color) for color in colors])
        hsv_colors[:, 1] *= 1.4
        hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)
        adjusted_colors = np.array([colorsys.hsv_to_rgb(*color) for color in hsv_colors])
        pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(coor_pred)
        opt = vis.get_render_option()
        opt.point_size = 10.0

        vis.run()
        vis.destroy_window()
    
if __name__ == "__main__":
    demo_path = "/home/user/yzchen_ws/imitation_learning/RiEMann/data/mug/pick/demo.npz"
    main(demo_path)
