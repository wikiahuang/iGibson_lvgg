"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import logging
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import pybullet as p
import cv2
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as transforms

from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from floorplan_verse.model.all_in_one_net import all_in_one

action_space = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "STOP"}

def updata_state(s: Simulator, robot: ManipulationRobot, history_rgb: torch.Tensor, history_depth: torch.Tensor, history_pose: torch.Tensor):
    cur_frames= s.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
    cur_rgb = cur_frames[0]
    cur_depth = cur_frames[1]
    cur_rgb = Image.fromarray((255 * cur_rgb[:, :, :3]).astype(np.uint8))
    cur_rgb = transforms.ToTensor()(cur_rgb).unsqueeze(0).to("cuda:0")  # 1, C, H, W
    cur_depth = np.linalg.norm(cur_depth[:, :, :3], axis=2)
    cur_depth = np.clip(cur_depth, None, 10) * 25.5
    cur_depth = cur_depth.astype(np.uint8)
    cur_depth = Image.fromarray(cur_depth)
    cur_depth = transforms.ToTensor()(cur_depth).unsqueeze(0).to("cuda:0")
    # update history
    if history_rgb.shape[1] == 8:
        history_rgb = torch.cat([history_rgb[:, 1:, :, :, :], cur_rgb.unsqueeze(1)], dim=1)
        history_depth = torch.cat([history_depth[:, 1:, :, :, :], cur_depth.unsqueeze(1)], dim=1)
        history_pose = torch.cat(
            [history_pose[:, 1:, :], 
            torch.from_numpy(robot.get_position()[:2]).unsqueeze(0).unsqueeze(0).to("cuda:0").float()], 
            dim=1
        )
    else:
        history_rgb = torch.cat([history_rgb, cur_rgb.unsqueeze(1)], dim=1)
        history_depth = torch.cat([history_depth, cur_depth.unsqueeze(1)], dim=1)
        history_pose = torch.cat(
            [history_pose, 
            torch.from_numpy(robot.get_position()[:2]).unsqueeze(0).unsqueeze(0).to("cuda:0").float()], 
            dim=1
        )

def action_to_control(action: list):
    """
    convert discret actions to continuous control
    action: list of discret actions
    """
    controls = []
    for a in action:
        if a == 0:
            controls.append([0.4, 0.0])  # forward
        elif a == 1:
            controls.append([0.0, 0.18])  # left
        elif a == 2:
            controls.append([0.0, -0.18])  # right
        else:
            controls.append([0.0, 0.0])  # stop
    return controls

def waypoint_to_action(waypoints: np.array):
    """
    convert waypoints to discret actions
    waypoints: local 2D coordinates, shape (N, 2)
    """
    cur_yaw = 0  # facing +x direction
    last_x, last_y = 0.0, 0.0
    actions = []
    for waypoint in waypoints:
        next_yaw = np.arctan2(waypoint[1] - last_y, waypoint[0] - last_x)
        yaw_diff = next_yaw - cur_yaw
        # forward
        if np.abs(yaw_diff) < (5 * np.pi / 180):
            actions.append(0)
        # left
        elif yaw_diff >= (5 * np.pi / 180):
            turns = int(yaw_diff / (15 * np.pi / 180)) + 1
            for _ in range(turns):
                actions.append(1)
        # right
        else:
            turns = int(-yaw_diff / (15 * np.pi / 180)) + 1
            for _ in range(turns):
                actions.append(2)
        last_x, last_y = waypoint[0], waypoint[1]
        cur_yaw = next_yaw
    return actions
def yaw_rotmat(yaw):
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_local_coords(positions, curr_pos, curr_yaw):
    """
    Convert positions to local coordinates

    Args:
        positions (list): positions to convert
        curr_pos (list): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    positions = np.array(positions)
    print("positions shape: {}".format(positions.shape))
    print("position: {}".format(positions))  
    curr_pos = np.array(curr_pos)
    curr_yaw = float(curr_yaw)
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def main(selection="user", short_exec=False):
    """
    evaluate floverse model
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    render_mode, use_pb_gui = "gui_interactive", False
    # render_mode, use_pb_gui = "headless", False
    s = Simulator(mode=render_mode, use_pb_gui=use_pb_gui, image_width=512, image_height=512)
    scene = StaticIndoorScene(
        "00025-ixTj1aTMup2",
        build_graph=True,
    )
    s.import_scene(scene)
    robot = REGISTERED_ROBOTS["Locobot"](
        action_type="continuous",
        action_normalize=True,
        controller_config={
            'base': {'name': 'DifferentialDriveController'}
        },
    )
    s.import_object(robot)

    # init model 
    config = OmegaConf.load('/home/user/data/vis_nav/iGibson/floorplan_verse/config/all_in_one.yaml')
    policy = all_in_one(config).to("cuda:0")
    # load ckpt
    ckpt = torch.load('/home/user/data/vis_nav/iGibson/floorplan_verse/wandb/latest-run/files/model_epoch_10.pth', map_location='cuda:0') 
    policy.load_state_dict(ckpt)
    policy.eval()

    # prepare dataset inputs
    floorplan_path = "/home/user/data/vis_nav/iGibson/igibson/data/g_dataset/00031-Wo6kuutE9i7/denoise_after_dilate_0.png"
    floorplan = Image.open(floorplan_path)
    floorplan = transforms.ToTensor()(floorplan)

  

    # -2.700000 -0.100000 -1.700000 -0.100000
    # -2.600000 -0.100000 -1.600000 -0.100000
    # -2.500000 -0.100000 -1.500000 -0.100000
    # -2.400000 -0.100000 -1.400000 -0.100000
    # -2.300000 -0.100000 -1.300000 -0.100000
    # -2.200000 -0.100000 -1.200000 -0.100000
    # -2.100000 -0.100000 -1.100000 -0.100000
    # -2.000000 0.000000 -1.292894 0.707107
    # -1.900000 0.100000 -1.192893 0.807107
    current_position = np.array([-1.900000, 0.100000])
    point_goal =  np.array([2.000000, -0.400000])
    history_position = np.array(
        [[-2.700000, -0.100000],
        [-2.600000, -0.100000],
        [-2.500000, -0.100000],
        [-2.400000, -0.100000],
        [-2.300000, -0.100000],
        [-2.200000, -0.100000],
        [-2.100000, -0.100000],
        [-2.000000, 0.000000]]
    )
    history_orientation_point = np.array(
        [[-1.700000, -0.100000],
        [-1.600000, -0.100000],
        [-1.500000, -0.100000],
        [-1.400000, -0.100000],
        [-1.300000, -0.100000],
        [-1.200000, -0.100000],
        [-1.100000, -0.100000],
        [-1.292894, 0.707107]]  
    )
    history_orientation_yaw = np.arctan2(
        history_orientation_point[:, 1] - history_position[:, 1],
        history_orientation_point[:, 0] - history_position[:, 0]
    )
    history_quaternion = np.array(
        [p.getQuaternionFromEuler([0, 0, yaw]) for yaw in history_orientation_yaw]
    )
    # set robot in each history pose and get rgb, depth
    history_rgb_list = []
    history_depth_list = []
    step = 0
    for pose, ori in zip(history_position, history_quaternion):
        robot.set_position([pose[0], pose[1], 0])
        robot.set_orientation(ori)
        # robot.reset()
        # robot.keep_still()
        s.step()
        frames= s.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
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
        time.sleep(0.1)
        step += 1
    history_rgb = torch.cat(history_rgb_list, dim=0).unsqueeze(0).to("cuda:0")  # 1, T, C, H, W
    history_depth = torch.cat(history_depth_list, dim=0).unsqueeze(0).to("cuda:0")  # 1, T, 1, H, W

    floorplan = floorplan.unsqueeze(0).to("cuda:0")  # 1, C, H, W
    history_pose = torch.from_numpy(history_position).unsqueeze(0).to("cuda:0").float()  # 1, T, 2
    relative_point_goal = to_local_coords(point_goal.tolist(), current_position.tolist(), history_orientation_yaw[-1])
    relative_point_goal = torch.tensor(relative_point_goal).unsqueeze(0).to("cuda:0").float()  # 1, 2
    last_position = current_position.copy()
    while np.linalg.norm(current_position - point_goal) > 1:
        with torch.no_grad():
            action = policy.inference(
                history_rgb,
                history_depth,
                history_pose,
                floorplan,
                None,
                relative_point_goal,
                None
            )
        actions = action.squeeze(0).cpu().numpy()
        # print("predicted action: {}".format(actions))
        actions = waypoint_to_action(actions[1:7])
        for action in actions:
            if action == 0:
                print("take action: {}".format(action_space[action]))
                #forward
                robot.apply_action(np.array([0.2, 0.0]))
                for i in range(71):
                    s.step()
                updata_state(s, robot, history_rgb, history_depth, history_pose)
            elif action == 1:
                print("take action: {}".format(action_space[action]))
                # left
                robot.apply_action(np.array([0.0, -0.18]))
                for i in range(10):
                    s.step()
                updata_state(s, robot, history_rgb, history_depth, history_pose)
            elif action == 2:
                print("take action: {}".format(action_space[action]))
                # right
                robot.apply_action(np.array([0.0, 0.18]))
                for i in range(10):
                    s.step()
                updata_state(s, robot, history_rgb, history_depth, history_pose)
            elif action == 3:
                # stop
                pass
        current_position = robot.get_position()[:2]
        current_yaw = robot.get_rpy()[2]
        if np.linalg.norm(current_position - last_position) < 0.05:
            print("robot is stuck, turning left")
            robot.apply_action(np.array([0.0, -0.18]))
            for i in range(8):
                for i in range(10):
                    s.step()
                updata_state(s, robot, history_rgb, history_depth, history_pose)    
            current_position = robot.get_position()[:2]
            current_yaw = robot.get_rpy()[2]    
        last_position = current_position.copy() 
        relative_point_goal = to_local_coords(point_goal.tolist(), current_position.tolist(), current_yaw)
        relative_point_goal = torch.tensor(relative_point_goal).unsqueeze(0).to("cuda:0").float()  # 1, 2
        print("current_position: {}, point_goal: {}, distance: {}".format(current_position, point_goal, np.linalg.norm(current_position - point_goal)))
                







    # Set initial viewer if using IG GUI
    # gui = "ig"
    # if gui != "pb" and not headless:
    #     s.viewer.initial_pos = [1.6, 0, 1.3]
    #     s.viewer.initial_view_direction = [-0.7, 0, -0.7]
    #     s.viewer.reset_viewer()



    # cur_pos = robot.get_position()
    # cur_ori = robot.get_orientation()
    # cur_yaw = robot.get_rpy()[2]
    # print("Initial robot position: {}".format(cur_pos))
    # print("Initial robot orientation: {}".format(cur_ori))
    # step = 0

    # while step != max_steps:
    #     # prepare inputs
    #     rgb, depth = robot.get_camera_observation(visualize=False, get_rgb=True, get_depth=True)
    #     rgb = Image.fromarray((255 * rgb).astype(np.uint8))
    #     depth = Image.fromarray((depth * 1000).astype(np.uint16))
    # while True:
    #     robot.apply_action(np.array([0.4, 0.0]))
    #     for i in range(71):
    #         s.step()    

    #     print("move_distance: {}".format(np.linalg.norm(robot.get_position() - cur_pos)))   
    #     cur_pos = robot.get_position()     
    # after_pos = robot.get_position()
    # while True:
    #     robot.apply_action(np.array([0, 0.18]))
    #     for i in range(10):
    #         s.step()
    #     frames = s.renderer.render_robot_cameras(modes=("rgb"), cache=False)
    #     img = Image.fromarray((255 * np.concatenate(frames, axis=1)[:, :, :3]).astype(np.uint8))
    #     img.save("/home/user/data/weiqi/turn_right_frame_{}.png".format(step))
    #     print("turning angle: {}".format((robot.get_rpy()[2] - cur_yaw) * (180 / np.pi)))
    #     # time.sleep(0.5)
    #     cur_yaw = robot.get_rpy()[2]
    #     step += 1

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
