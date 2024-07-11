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



def gen_grasps(input_path, output_path):

    # cfg_path = os.path.join("config", "real_world", "config.json")
    cfg_path = "/home/xuhang/Desktop/yzchen_ws/RiEMann_seg/config/real_world/config.json"
    all_cfg = OmegaConf.load(cfg_path)
    cfg_mani = all_cfg.mani

    # model_path = os.path.join("experiments", "real_world", "maninet.pth")
    model_path = "/home/xuhang/Desktop/yzchen_ws/RiEMann_seg/experiments/real_world/maninet.pth"
    policy_mani = globals()[cfg_mani.model](voxel_size=cfg_mani.voxel_size, \
                                                    radius_threshold=cfg_mani.radius_threshold).float().to(cfg_mani.device)
    policy_mani.load_state_dict(torch.load(model_path))
    policy_mani.eval()
    
    # load data
    input_data = np.load(input_path)
    try_number = input_data['try_number']
    cpu_xyz = input_data['xyz']
    cpu_rgb = input_data['rgb']

    obj_center= input_data['pc_center']


    input_xyz = torch.tensor(cpu_xyz).float().unsqueeze(0).to(cfg_mani.device)
    input_rgb = torch.tensor(cpu_rgb).float().unsqueeze(0).to(cfg_mani.device)
    
    # preprossing input to keep the same as training
    data = {
        "xyz": input_xyz[0],
        "rgb": input_rgb[0],
    }
    # data = downsample_table(data)

    xyz = data["xyz"].unsqueeze(0).to(cfg_mani.device)
    rgb = data["rgb"].unsqueeze(0).to(cfg_mani.device)

    pred_pos_list = []
    pred_rot_list = []
    with torch.no_grad():
        for i in range(try_number):
            ref_point = torch.tensor(obj_center).float().unsqueeze(0).to(cfg_mani.device)

            output_pos, output_direction = policy_mani(
                {"xyz": xyz, "rgb": rgb}, 
                reference_point=ref_point, 
                distance_threshold=cfg_mani.distance_threshold,
                save_ori_feature = True,
                draw_pcd=True,
                # pcd_name=args.setting,  # required when draw_pcd == True
            )
            out_dir_schmidt = modified_gram_schmidt(output_direction.reshape(-1, 3).T, to_cuda=True)



            pred_pos = output_pos.detach().cpu().numpy().reshape(3)
            pred_rot = out_dir_schmidt.detach().cpu().numpy()

            pred_pos_list.append(pred_pos)
            pred_rot_list.append(pred_rot)
    
    pred_pos_arr = np.asarray(pred_pos_list)
    pred_rot_arr = np.asarray(pred_rot_list)
    np.savez(output_path, pred_pos=pred_pos_arr, pred_rot=pred_rot_arr)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz_path", type=str, default="/home/xuhang/Desktop/yzchen_ws/RiEMann_seg/experiments/real_world/input.npz")
    parser.add_argument("--output_npz_path", type=str, default="/home/xuhang/Desktop/yzchen_ws/RiEMann_seg/experiments/real_world/output.npz")
    args = parser.parse_args()
    input_npz_path = args.input_npz_path
    output_npz_path = args.output_npz_path  
    gen_grasps(input_npz_path, output_npz_path)




