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
from multiprocessing import Pool, set_start_method

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import igibson
import pybullet as p
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets, download_demo_data
from igibson.examples.scripts.utils import  updata_state,  to_local_coords, to_global_coords, get_pose_from_position
from igibson.simulator import Simulator
from igibson.robots.manipulation_robot import ManipulationRobot
import torchvision.transforms.functional as TF

metric_waipoint_spacing = 0.045
waypoint_spacing = 1
ACTION_STATS = {
    "min": np.array([-2.5, -4.0]),
    "max": np.array([5.0, 4.0]),
}
aspect_ratio = (
    4 / 3
) 
transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
transform = transforms.Compose(transform)

def updata_state_nomad(s: Simulator, robot: ManipulationRobot, history_rgb: torch.Tensor, history_global_position: np.array):
    cur_frames= s.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
    cur_rgb = cur_frames[0]
    cur_depth = cur_frames[1]
    cur_rgb = Image.fromarray((255 * cur_rgb[:, :, :3]).astype(np.uint8))
    cur_rgb = cur_rgb.resize((96, 96))
    cur_rgb = transforms.ToTensor()(cur_rgb).unsqueeze(0).to("cuda:0")  # 1, C, H, W

    # update history
    if history_rgb.shape[1] == 4:
        history_rgb = torch.cat([history_rgb[:, 3:, :, :], cur_rgb], dim=1)
        history_global_position = np.concatenate([history_global_position[1:, :], robot.get_position()[:2].reshape(1, 2)],axis=0)
    else:
        history_rgb = torch.cat([history_rgb, cur_rgb], dim=1)
        history_global_position = np.concatenate([history_global_position[1:, :], robot.get_position()[:2].reshape(1, 2)],axis=0) 

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)

def load_model(eval_config):
    """
    load floverse model
    """
    config = yaml.load(open(eval_config["model_config"], "r"), Loader=yaml.FullLoader)
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
                
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    policy = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    ).to("cuda:0")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    # load ckpt
    ckpt = torch.load(config["ckpt_path"], map_location='cuda:0')
    policy.load_state_dict(ckpt, strict=False)
    policy.eval()
    return policy, noise_scheduler   

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
    w, h = image_goal.size
    image_goal = TF.center_crop(image_goal, (int(w / aspect_ratio), w))
    img = image_goal.resize((96, 96))
    resize_img = TF.to_tensor(img).to("cuda:0")  # 1, C, H, W
    image_goal = transform(resize_img).unsqueeze(0).to("cuda:0")  # 1, C, H, W
    point_goal = traj_data[-1, :2]

    current_position = traj_data[3, :2]
    history_global_position = traj_data[:4, :2]
    history_global_orientation_point = traj_data[:4, 2:4]
    history_orientation_yaw = np.arctan2(
        history_global_orientation_point[:, 1] - history_global_position[:, 1],
        history_global_orientation_point[:, 0] - history_global_position[:, 0]
    )
    history_quaternion = np.array(
        [p.getQuaternionFromEuler([0, 0, yaw]) for yaw in history_orientation_yaw]
    )
        
    history_rgb_list = []
    for pose, ori in zip(history_global_position, history_quaternion):
        env.robots[0].set_position([pose[0], pose[1], 0])
        env.robots[0].set_orientation(ori)
        env.simulator_step()
        frames= env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)
        rgb = frames[0]
        rgb = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
        w, h = rgb.size
        rgb = TF.center_crop(rgb, (int(w / aspect_ratio), w))
        rgb = rgb.resize((96, 96))
        rgb = TF.to_tensor(rgb).to("cuda:0")  # 1, C, H, W
        rgb = transform(rgb)
        history_rgb_list.append(rgb)
    env.simulator_step()    
    history_rgb = torch.cat(history_rgb_list, dim=0).unsqueeze(0).to("cuda:0")  # 1, T*C, H, W
    return history_rgb,  history_global_position, current_position, point_goal, image_goal

def eval_scene(scene_id, eval_config):
    policy, scheduler = load_model(eval_config)
    scenes_dir = eval_config["scenes_dir"]
    traj_save_dir = eval_config["traj_save_dir"]
    traj_save_scene_dir = os.path.join(traj_save_dir, scene_id)
    os.makedirs(traj_save_scene_dir, exist_ok=True)
    scene_name = scene_id.split("_")[0] 
    floor = int(scene_id.split("_")[1])
    config_filename = os.path.join(igibson.configs_path, "locobot_floverse_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False
    config_data["scene_id"] = scene_name
    env = iGibsonEnv(config_file=config_data, mode="headless")
    trajs = [t for t in os.listdir(os.path.join(scenes_dir, scene_id)) if os.path.isdir(os.path.join(scenes_dir, scene_id, t))]
    trajs = sorted(trajs, key=lambda x: int(re.search(r'\d+', x).group()))
    # print("trajs to eval:", trajs[:30]) 
    for traj_id in trajs[:10]:
        if not os.path.isdir(os.path.join(scenes_dir, scene_id, traj_id)):
            continue
        print(f"[PID {os.getpid()}] Evaluating scene {scene_id}, traj {traj_id}")
        traj_save_id_dir = os.path.join(traj_save_scene_dir, traj_id)
        os.makedirs(traj_save_id_dir, exist_ok=True)
        traj_save_id_dir = os.path.join(traj_save_scene_dir, traj_id)   
        if not os.path.exists(traj_save_id_dir):
            os.makedirs(traj_save_id_dir)
        traj_npy = []
        collisions = 0
        steps = 0
        history_rgb, history_global_position, current_position, point_goal, image_goal = init_states(env, scenes_dir, scene_id, traj_id)
        last_position = current_position.copy()
        while np.linalg.norm(current_position - point_goal) > 1 and steps < 1000:
            with torch.no_grad():
                no_mask = torch.zeros((history_rgb.shape[0],)).long().to("cuda:0")
                obsgoal_cond = policy("vision_encoder", obs_img=history_rgb, goal_img=image_goal, input_goal_mask=no_mask)
                obsgoal_cond = obsgoal_cond.repeat_interleave(10, dim=0)
                noisy_diffusion_output = torch.randn(
                    (len(obsgoal_cond), 8, 2), device="cuda:0")
                diffusion_output = noisy_diffusion_output
                for k in scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = policy(
                        "noise_pred_net",
                        sample=diffusion_output,
                        timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to("cuda:0"),
                        global_cond=obsgoal_cond
                    )

                    # inverse diffusion step (remove noise)
                    diffusion_output = scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=diffusion_output
                    ).prev_sample
                diffusion_output = diffusion_output.mean(dim=0, keepdim=True)
                gc_actions = get_action(diffusion_output, ACTION_STATS)
                actions = gc_actions.squeeze(0).cpu().numpy()
                actions = actions * waypoint_spacing * metric_waipoint_spacing 
                # print("Predicted actions:", actions)
            current_position = env.robots[0].get_position()[:2]
            current_yaw = env.robots[0].get_rpy()[2]
            predict_positions = to_global_coords(actions[:4].tolist(), current_position.tolist(), current_yaw) 
            predict_pose, yaws = get_pose_from_position(predict_positions)
            for i in range(len(predict_pose)):
                position_3d = np.array([predict_pose[i, 0], predict_pose[i, 1], 0.1])
                euler = np.array([0, 0, yaws[i]])
                valid = env.test_valid_position(env.robots[0], position_3d, euler, ignore_self_collision=True)
                if not valid:
                    # print("Collition at predicted step {}, position: {}, orientation: {}".format(i, position_3d, euler))
                    # compute the shortest path to the goal 
                    safe_path, _ , _= env.simulator.scene.get_shortest_path(floor, env.robots[0].get_position()[:2], point_goal, entire_path=True)
                    next_position = safe_path[1]  
                    next_ori = safe_path[2] - safe_path[1]
                    next_yaw = np.arctan2(next_ori[1], next_ori[0])
                    next_quat = p.getQuaternionFromEuler([0, 0, next_yaw])
                    env.robots[0].set_position([next_position[0], next_position[1], 0])
                    env.robots[0].set_orientation(next_quat)
                    env.simulator.step()
                    # time.sleep(1)
                    for _update in range(8):
                        updata_state_nomad(env.simulator, env.robots[0], history_rgb, history_global_position)
                    current_rgb = env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)[0]
                    current_rgb = Image.fromarray((255 * current_rgb[:, :, :3]).astype(np.uint8))
                    current_rgb = current_rgb.resize((96, 96))    
                    current_rgb.save(os.path.join(traj_save_id_dir, "step_{}.png".format(steps)))   
                    collisions += 1
                    steps += 1
                    traj_npy.append(np.array([next_position[0], next_position[1], next_yaw, 1]))
                    break
                next_position = predict_pose[i, :2]
                next_quat = predict_pose[i, 2:]
                env.robots[0].set_position([next_position[0], next_position[1], 0])
                env.robots[0].set_orientation(next_quat)                   
                env.simulator.step()
                updata_state_nomad(env.simulator, env.robots[0], history_rgb, history_global_position)
                current_rgb = env.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)[0]
                current_rgb = Image.fromarray((255 * current_rgb[:, :, :3]).astype(np.uint8))
                current_rgb = current_rgb.resize((96, 96))
                current_rgb.save(os.path.join(traj_save_id_dir, "step_{}.png".format(steps)))
                steps += 1
                # time.sleep(1)
                traj_npy.append(np.array([next_position[0], next_position[1], yaws[i], 0])) # 0 for no collision
            current_position = env.robots[0].get_position()[:2]
            current_yaw = env.robots[0].get_rpy()[2]
        np.savetxt(os.path.join(traj_save_id_dir, "{}.txt".format(traj_id)), np.array(traj_npy))  
        print("Finished scene {}, traj {}, steps {}, collisions {}".format(scene_id, traj_id, steps, collisions))
    env.close()

def main(selection="user", headless=True, short_exec=False, eval_config=None):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs (not interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    scenes_dir = eval_config["scenes_dir"]
    scenes_ids = [
        scene for scene in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, scene))
    ]
    
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
        "model_config": "/home/weiqi/data/iGibson/visualnav_transformer/train/config/nomad.yaml",
        "traj_save_dir": "/home/weiqi/data/iGibson/igibson/examples/scripts/results/eval_nomad",
    }
    main(eval_config=eval_config)
