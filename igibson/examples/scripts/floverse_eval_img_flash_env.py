import logging
import os
from sys import platform
import time

import yaml
import re
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
from multiprocessing import Pool, set_start_method


sys.path.append('/media/data/weiqi_data/code')
import floorplan_verse
from floorplan_verse.model.all_in_one_net import all_in_one
import igibson
from igibson.utils.assets_utils import download_assets, download_demo_data
import pybullet as p
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.examples.scripts.utils import  updata_state, to_local_coords, to_global_coords, get_pose_from_position

def load_model(eval_config):
    """
    load floverse model
    """
    config = OmegaConf.load(eval_config["model_config"])
    policy = all_in_one(config).to("cuda:0")
    # load ckpt
    ckpt = torch.load(config["ckpt_path"], map_location='cuda:0')
    for key in list(ckpt.keys()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            ckpt[new_key] = ckpt.pop(key)
    policy.load_state_dict(ckpt)
    policy.eval()
    return policy   

def init_states(env, scenes_dir, scene_id, traj_id):
    """
    initialize the states for floverse eval
    """
    traj_data = np.loadtxt(os.path.join(scenes_dir, scene_id, traj_id, "{}.txt".format(traj_id)))
    goal_position = traj_data[-1, :2]
    goal_orientation = traj_data[-1, 2:4] - traj_data[-1, :2]
    goal_yaw = np.arctan2(goal_orientation[1], goal_orientation[0])
    goal_quat = p.getQuaternionFromEuler([0, 0, goal_yaw])
    env.robots[0].set_position([goal_position[0], goal_position[1], 0])
    env.robots[0].set_orientation(goal_quat)
    env.simulator_step()
    frames= env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)
    image_goal = frames[0]
    image_goal = Image.fromarray((255 * image_goal[:, :, :3]).astype(np.uint8))
    image_goal = transforms.ToTensor()(image_goal).unsqueeze(0).to("cuda:0")  # 1, C, H, W

    current_position = traj_data[7, :2]
    history_global_position = traj_data[:8, :2]
    history_global_orientation_point = traj_data[:8, 2:4]
    history_orientation_yaw = np.arctan2(
        history_global_orientation_point[:, 1] - history_global_position[:, 1],
        history_global_orientation_point[:, 0] - history_global_position[:, 0]
    )
    history_quaternion = np.array(
        [p.getQuaternionFromEuler([0, 0, yaw]) for yaw in history_orientation_yaw]
    )
    history_position = to_local_coords(history_global_position.tolist(), current_position.tolist(), history_orientation_yaw[-1])         
    history_rgb_list = []
    history_depth_list = []
    for pose, ori in zip(history_global_position, history_quaternion):
        env.robots[0].set_position([pose[0], pose[1], 0])
        env.robots[0].set_orientation(ori)
        env.simulator_step()
        frames= env.simulator.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
        rgb = frames[0]
        depth = frames[1]
        rgb = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
        rgb = transforms.ToTensor()(rgb)
        history_rgb_list.append(rgb.unsqueeze(0))
        depth = np.linalg.norm(depth[:, :, :3], axis=2)
        depth = np.clip(depth, None, 10) * 25.5
        depth = depth.astype(np.uint8)
        depth = Image.fromarray(depth)
        depth = transforms.ToTensor()(depth)    
        history_depth_list.append(depth.unsqueeze(0))
    env.simulator_step()    
    history_rgb = torch.cat(history_rgb_list, dim=0).unsqueeze(0).to("cuda:0")  # 1, T, C, H, W
    history_depth = torch.cat(history_depth_list, dim=0).unsqueeze(0).to("cuda:0")  # 1, T, 1, H, W
    history_pose = torch.from_numpy(history_position).unsqueeze(0).to("cuda:0").float()  # 1, T, 2
    return history_rgb, history_depth, history_pose, history_global_position, current_position, image_goal, goal_position


def eval_scene(scene_id, eval_config):
    torch.cuda.set_device(0)
    print(f"[PID {os.getpid()}] Evaluating {scene_id}")

    policy = load_model(eval_config)
    scenes_dir = eval_config["scenes_dir"]
    traj_save_scene_dir = os.path.join(eval_config["traj_save_dir"], scene_id)
    os.makedirs(traj_save_scene_dir, exist_ok=True)

    scene_name = scene_id.split("_")[0]
    floor = int(scene_id.split("_")[1])
    config_filename = os.path.join(igibson.configs_path, "locobot_floverse_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False
    config_data["scene_id"] = scene_name
    env = iGibsonEnv(config_file=config_data, mode="headless")

    floorplan = Image.open(os.path.join(scenes_dir, scene_id, "floorplan.png"))
    floorplan = transforms.ToTensor()(floorplan).unsqueeze(0).to("cuda:0")

    trajs = [t for t in os.listdir(os.path.join(scenes_dir, scene_id)) if os.path.isdir(os.path.join(scenes_dir, scene_id, t))]
    trajs = sorted(trajs, key=lambda x: int(re.search(r'\d+', x).group()))
    print("trajs to eval:", trajs[:30]) 
    for traj_id in trajs[:30]:
        start_time = time.time()
        print(f"---Evaluating {scene_id}/{traj_id}")
        traj_save_id_dir = os.path.join(traj_save_scene_dir, traj_id)
        os.makedirs(traj_save_id_dir, exist_ok=True)
        traj_npy, collisions, steps = [], 0, 0

        history_rgb, history_depth, history_pose, history_global_position, current_position, image_goal, point_goal = init_states(env, scenes_dir, scene_id, traj_id)

        while np.linalg.norm(current_position - point_goal) > 1 and steps < 500:
            with torch.no_grad():
                actions = policy.inference(history_rgb, history_depth, history_pose, floorplan, image_goal, None, None)
                actions = actions.squeeze(0).cpu().numpy()
            current_position = env.robots[0].get_position()[:2]
            current_yaw = env.robots[0].get_rpy()[2]
            predict_positions = to_global_coords(actions[:16].tolist(), current_position.tolist(), current_yaw)
            predict_pose, yaws = get_pose_from_position(predict_positions)
            for i in range(len(predict_pose)):
                position_3d = np.array([predict_pose[i, 0], predict_pose[i, 1], 0.1])
                euler = np.array([0, 0, yaws[i]])
                valid = env.test_valid_position(env.robots[0], position_3d, euler, ignore_self_collision=True)
                if not valid:
                    safe_path, _, _ = env.simulator.scene.get_shortest_path(floor, env.robots[0].get_position()[:2], point_goal, entire_path=True)
                    if len(safe_path) < 1:
                        break
                    next_position = safe_path[0]
                    next_ori = safe_path[1] - safe_path[0]
                    next_yaw = np.arctan2(next_ori[1], next_ori[0])
                    next_quat = p.getQuaternionFromEuler([0, 0, next_yaw])
                    env.robots[0].set_position([next_position[0], next_position[1], 0])
                    env.robots[0].set_orientation(next_quat)
                    env.simulator.step()
                    for _update in range(8):
                        updata_state(env.simulator, env.robots[0], history_rgb, history_depth, history_pose, history_global_position)
                    current_rgb = env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)[0]
                    current_rgb = Image.fromarray((255 * current_rgb[:, :, :3]).astype(np.uint8))
                    current_rgb = current_rgb.resize((96, 96))
                    current_rgb.save(os.path.join(traj_save_id_dir, f"step_{steps}.png"))
                    collisions += 1
                    steps += 1
                    traj_npy.append(np.array([next_position[0], next_position[1], next_yaw, 1]))
                    break
                next_position = predict_pose[i, :2]
                next_quat = predict_pose[i, 2:]
                env.robots[0].set_position([next_position[0], next_position[1], 0])
                env.robots[0].set_orientation(next_quat)
                env.simulator.step()
                updata_state(env.simulator, env.robots[0], history_rgb, history_depth, history_pose, history_global_position)
                current_rgb = env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)[0]
                current_rgb = Image.fromarray((255 * current_rgb[:, :, :3]).astype(np.uint8))
                current_rgb = current_rgb.resize((96, 96))
                current_rgb.save(os.path.join(traj_save_id_dir, f"step_{steps}.png"))
                steps += 1
                traj_npy.append(np.array([next_position[0], next_position[1], yaws[i], 0]))
            current_position = env.robots[0].get_position()[:2]
            current_yaw = env.robots[0].get_rpy()[2]

        np.savetxt(os.path.join(traj_save_id_dir, f"{traj_id}.txt"), np.array(traj_npy))
        print(f"Finished {scene_id}/{traj_id}, steps={steps}, collisions={collisions}, time={time.time()-start_time:.2f}s")

    env.close()


def main(eval_config):
    download_assets()
    download_demo_data()
    scenes_dir = eval_config["scenes_dir"]
    scenes_ids = [s for s in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, s))]

    nproc = min(len(scenes_ids), 4)  
    print(f"Launching {nproc} parallel processes...")

    with Pool(nproc) as pool:
        pool.starmap(eval_scene, [(scene_id, eval_config) for scene_id in scenes_ids])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    eval_config = {
        "scenes_dir": "/home/weiqi/data/floverse_data/eval/eval_with_obj",
        "model_config": "/home/weiqi/code/floorplan-verse/config/all_in_one.yaml",
        "traj_save_dir": "/home/weiqi/data/iGibson/igibson/examples/scripts/results/eval_floverse_image_without_refiner",
    }
    main(eval_config=eval_config)
