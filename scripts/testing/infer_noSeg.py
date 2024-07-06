import sys
sys.path.append(".")
import os
import torch
from networks import *
from omegaconf import OmegaConf
from utils.data_utils import downsample_table
import argparse
import numpy as np
from utils.utils import modified_gram_schmidt

def main(args):
    all_cfg = OmegaConf.load(f"config/{args.exp_name}/{args.pick_or_place}/config.json")
    cfg_seg = all_cfg.seg
    cfg_mani = all_cfg.mani

    wd = os.path.join("experiments", args.exp_name, args.pick_or_place)
    pcd_path = os.path.join("data", args.exp_name, args.pick_or_place, f"{args.setting}.npz")

    # see if one instance in the trainning set is ok
    select_id = 3
    demo = np.load(pcd_path)
    cpu_xyz = demo['xyz'][select_id]
    cpu_rgb = demo['rgb'][select_id]
    if cfg_mani.ref_point == "gt":
        ref_point_cpu = demo['seg_center'][select_id]
    elif cfg_mani.ref_point == "center":
        ref_point_cpu = cpu_xyz.mean(axis=0)  #  Note: cpu_xyz.shape = (1024, 3)
    np.savez('/home/user/yzchen_ws/imitation_learning/RiEMann/data/mug/pick/data{}.npz'.format(select_id), xyz = cpu_xyz, rgb = cpu_rgb)

    input_xyz = torch.tensor(cpu_xyz).float().unsqueeze(0).to(cfg_seg.device)
    input_rgb = torch.tensor(cpu_rgb).float().unsqueeze(0).to(cfg_seg.device)
    
    model_dir = os.path.join("experiments", args.exp_name, args.pick_or_place)
    # policy_seg = globals()[cfg_seg.model](voxel_size=cfg_seg.voxel_size, radius_threshold=cfg_seg.radius_threshold).float().to(cfg_seg.device)
    # policy_seg.load_state_dict(torch.load(os.path.join(model_dir, "segnet.pth")))
    # policy_seg.eval()

    policy_mani = globals()[cfg_mani.model](voxel_size=cfg_mani.voxel_size, radius_threshold=cfg_mani.radius_threshold).float().to(cfg_mani.device)
    policy_mani.load_state_dict(torch.load(os.path.join(model_dir, "maninet.pth")))
    policy_mani.eval()

    assert cfg_seg.device == cfg_mani.device, "Device mismatch between segmentation and manipulation networks!"

    # preprossing input to keep the same as training
    data = {
        "xyz": input_xyz[0],
        "rgb": input_rgb[0],
    }
    # data = downsample_table(data)

    xyz = data["xyz"].unsqueeze(0).to(cfg_seg.device)
    rgb = data["rgb"].unsqueeze(0).to(cfg_seg.device)

    with torch.no_grad():
        # ref_point = policy_seg(
        #     {"xyz": xyz, "rgb": rgb}, 
        #     draw_pcd=True,
        #     pcd_name=args.setting,
        # )

        ref_point = torch.tensor(ref_point_cpu).float().unsqueeze(0).to(cfg_seg.device)

        output_pos, output_direction = policy_mani(
            {"xyz": xyz, "rgb": rgb}, 
            reference_point=ref_point, 
            distance_threshold=cfg_mani.distance_threshold,
            save_ori_feature = True,
            draw_pcd=True,
            pcd_name=args.setting,
        )
        out_dir_schmidt = modified_gram_schmidt(output_direction.reshape(-1, 3).T, to_cuda=True)

    # ref_point_cpu is the center of the focus
    # ref_point_cpu = ref_point.detach().cpu().numpy()
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(ref_point_cpu)
    # o3d.io.write_point_cloud('ref_point.ply', pcd)

    pred_pos = output_pos.detach().cpu().numpy().reshape(3)
    pred_rot = out_dir_schmidt.detach().cpu().numpy()

    result_path = os.path.join(wd, f"{args.setting}_pred_pose.npz")
    np.savez(result_path,
             pred_pos=pred_pos,
             pred_rot=pred_rot,
            )
    print(f"Result saved to: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_name', type=str, default="mug")
    parser.add_argument('-pick_or_place', type=str, choices=["pick", "place"], default="pick")
    parser.add_argument('-setting', type=str, default='riemann_focus_demo')
    args = parser.parse_args()

    main(args)