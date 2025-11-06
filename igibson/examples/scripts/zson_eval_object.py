import logging
import os
from sys import platform
import time
import json
import sys
sys.path.append("/home/weiqi/code/floorplan-verse")
import yaml
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from baseline.zson.model.zson import ZSONPolicyNet
import igibson
import pybullet as p
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.assets_utils import download_assets, download_demo_data

action_space = {0:"STOP", 1:"MOVE_FORWARD", 2:"TURN_LEFT", 3: "TURN_RIGHT"}

def load_model(eval_config, device):
    """
    load zson model
    """
    config = OmegaConf.load(eval_config["model_config"])
    policy_params = config["model"]
    policy = ZSONPolicyNet(**policy_params).to(device)
    # load ckpt
    ckpt_dict = torch.load(config["ckpt_path"], map_location=device, weights_only=False)
    state_dict = ckpt_dict["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor_critic.net."):
            new_k = k.replace("actor_critic.net.", "", 1)
        elif k.startswith("actor_critic.action_distribution"):
            new_k = k.replace("actor_critic.action_distribution", "action_distribution", 1)
        elif k.startswith("actor_critic.critic"):
            new_k = k.replace("actor_critic.critic", "critic", 1)
        else:
            new_k = k
        new_state_dict[new_k] = v
    missing, unexpected = policy.load_state_dict(new_state_dict, strict=False)
    print("missing: ", missing)
    print("unexpected:", unexpected)
    print("finish initialize policy!")
    return policy, policy_params

def init_states(env, scenes_dir, scene_id, traj_id, device):
    """
    Initialize the robot's position and orientation
    """
    traj_data = np.loadtxt(os.path.join(scenes_dir, scene_id, traj_id, "{}.txt".format(traj_id)))
   
    current_position = traj_data[5, :2]
    current_ori_point = traj_data[5, 2:4]
    goal_position = traj_data[-1, :2]
    yaw = np.arctan2(
        current_ori_point[1] - current_position[1],
        current_ori_point[0] - current_position[0]
    )
    current_quaternion = p.getQuaternionFromEuler([0, 0, yaw])
    env.robots[0].set_position([current_position[0], current_position[1], 0])
    env.robots[0].set_orientation(current_quaternion)
    env.simulator_step()   
    frames= env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)
    rgb = frames[0]
    rgb = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
    rgb = transforms.ToTensor()(rgb).unsqueeze(0).permute(0, 2, 3, 1).to(device)
    return rgb, current_position, goal_position

def get_currrgb(s, device):
    cur_frames= s.renderer.render_robot_cameras(modes=("rgb"), cache=False)
    cur_rgb = cur_frames[0]
    cur_rgb_np = Image.fromarray((255 * cur_rgb[:, :, :3]).astype(np.uint8))
    cur_rgb = transforms.ToTensor()(cur_rgb_np).unsqueeze(0).permute(0, 2, 3, 1).to(device)  # 1, H, W, C
    return cur_rgb, cur_rgb_np

def main(selection="user", headless=True, short_exec=False, eval_config=None):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs (not interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # prepare scenes 
    scenes_dir = eval_config["scenes_dir"]
    scenes_ids = []
    for scene in os.listdir(scenes_dir):
        if os.path.isdir(os.path.join(scenes_dir, scene)):
            scenes_ids.append(scene)
    device = "cuda:0"
    policy, policy_param = load_model(eval_config, device)
    traj_save_dir = eval_config["traj_save_dir"]
    if not os.path.exists(traj_save_dir):
        os.makedirs(traj_save_dir)

    # evaluate on each scene
    for scene_id in scenes_ids:
        traj_save_scene_dir = os.path.join(traj_save_dir, scene_id)
        if not os.path.exists(traj_save_scene_dir):
            os.makedirs(traj_save_scene_dir)
        scene_name = scene_id.split("_")[0] 
        floor = int(scene_id.split("_")[1])
        config_filename = os.path.join(igibson.configs_path, "locobot_floverse_nav.yaml")
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        # Reduce texture scale for Mac.
        if platform == "darwin":
            config_data["texture_scale"] = 0.5
        # Shadows and PBR do not make much sense for a Gibson static mesh
        config_data["enable_shadow"] = False
        config_data["enable_pbr"] = False
        config_data["scene_id"] = scene_name  
        env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
       
        # evaluate on each traj
        for traj_id in os.listdir(os.path.join(scenes_dir, scene_id)):
            if not os.path.isdir(os.path.join(scenes_dir, scene_id, traj_id)):
                continue
            print("evaluating {}/{}".format(scene_id, traj_id))
            traj_save_id_dir = os.path.join(traj_save_scene_dir, traj_id)   
            if not os.path.exists(traj_save_id_dir):
                os.makedirs(traj_save_id_dir)

            traj_npy = []
            collisions = 0
            steps = 0
            num_env = 1
            action_shape = (1,)
            action_type = torch.long
            
            img, current_position, point_goal = init_states(env, scenes_dir, scene_id, traj_id, device)

            # load object goal
            json_dir = os.path.join(scenes_dir, scene_id, traj_id, "object", "object.json")
            with open(json_dir, "r") as f:
                object_data = json.load(f)  
            object_goal = object_data["object_category"] 

            recurrent_hidden_states = torch.zeros(num_env, policy.num_recurrent_layers, policy_param["hidden_size"], device=device)
            prev_actions = torch.ones(num_env , *action_shape, device=device, dtype=action_type)
            not_done_masks = torch.zeros(num_env, 1, device=device, dtype=torch.bool)
            while np.linalg.norm(current_position - point_goal) > 1 and steps < 500:
                with torch.no_grad():
                    value, action, action_log_probs, recurrent_hidden_states = policy(
                        img,
                        object_goal,
                        recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                    )
                    prev_actions.copy_(action)
                    action = action.squeeze(0).cpu().numpy()
                action = int(action)
                if action == 0:
                    print("take action: {}".format(action_space[action]))
                    # stop
                    break
                elif action == 1:
                    print("take action: {}".format(action_space[action]))
                    #forward
                    env.robots[0].apply_action(np.array([0.2, 0.0]))
                    for i in range(71):
                        env.simulator_step()   
                    img, img_np = get_currrgb(env.simulator, device)  
                elif action == 2:
                    print("take action: {}".format(action_space[action]))
                    # left
                    env.robots[0].apply_action(np.array([0.0, -0.18]))
                    for i in range(10):
                        env.simulator_step()   
                    img, img_np = get_currrgb(env.simulator, device)  
                elif action == 3:
                    print("take action: {}".format(action_space[action]))
                    # right
                    env.robots[0].apply_action(np.array([0.0, 0.18]))
                    for i in range(10):
                        env.simulator_step()   
                    img, img_np = get_currrgb(env.simulator, device)  

                current_position = env.robots[0].get_position()[:2]
                current_yaw = env.robots[0].get_rpy()[2]
                position_3d = np.array([current_position[0], current_position[1], 0.1])
                euler = np.array([0, 0, current_yaw])
                valid = env.test_valid_position(env.robots[0], position_3d, euler, ignore_self_collision=True)
                if not valid:
                    safe_path, _= env.simulator.scene.get_shortest_path(floor, env.robots[0].get_position()[:2], point_goal, entire_path=True)
                    if len(safe_path) < 1:
                        break
                    next_position = safe_path[0]
                    next_ori = safe_path[1] - safe_path[0]
                    next_yaw = np.arctan2(next_ori[1], next_ori[0])
                    next_quat = p.getQuaternionFromEuler([0, 0, next_yaw])
                    env.robots[0].set_position([next_position[0], next_position[1], 0])
                    env.robots[0].set_orientation(next_quat)
                    env.simulator.step()
                    img, img_np = get_currrgb(env.simulator, device)
                    img_np.save(os.path.join(traj_save_id_dir, f"step_{steps}.png"))
                    collisions += 1
                    steps += 1
                    traj_npy.append(np.array([next_position[0], next_position[1], next_yaw, np.linalg.norm(next_position - point_goal), 1]))
                    continue
                traj_npy.append(np.array([current_position[0], current_position[1], current_yaw, np.linalg.norm(current_position - point_goal), 0]))    
                img_np.save(os.path.join(traj_save_id_dir, "step_{}.png".format(steps)))
                steps += 1
            np.savetxt(os.path.join(traj_save_id_dir, "{}.txt".format(traj_id)), np.array(traj_npy))  
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_assets()
    download_demo_data()
    eval_config = yaml.load(open("/home/weiqi/data/iGibson/igibson/examples/scripts/config/zson.yaml", "r"), Loader=yaml.FullLoader)
    main(eval_config=eval_config)
