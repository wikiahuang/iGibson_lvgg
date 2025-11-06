import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import torch
from PIL import Image
import torchvision.transforms as transforms
import pybullet as p
from scipy.ndimage import distance_transform_edt

from distmap import (
    l1_distance_transform,
    l1_signed_transform,
    euclidean_distance_transform,
    euclidean_signed_transform,
)
from igibson.simulator import Simulator
from igibson.robots.manipulation_robot import ManipulationRobot

action_space = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "STOP"}
class Intrics:
    camera_fx = 128
    camera_fy = 128
    camera_cx = 128
    camera_cy = 128
intrics = Intrics()


def refine_path(occupancy_map, predicted_path, goal, alpha=1.0, beta=0.1, gamma=0.2, n_iter=5):
    """
    occupancy_map: 2D numpy array, 0=obstacle, 1=free
    predicted_path: (N, 2) array of waypoints
    goal: (2,)
    alpha:  weight for distance to obstacle
    beta:   weight for attraction to goal
    gamma: weight for smoothness
    n_iter: number of iterations
    """

    dist_map = distance_transform_edt(occupancy_map)
    max_dist = 1
    dist_map = dist_map / (max_dist + 1e-6)  
    safe_dist = 2.0
    h, w = occupancy_map.shape
    traj_pixels = predicted_path.copy()
    for i, (x, y) in enumerate(traj_pixels):
        pixel_x = int((y / 0.1) + w // 2)
        pixel_y = int(x / 0.1)
        traj_pixels[i] = [pixel_x, pixel_y]
    refined = traj_pixels.copy()

    goal = goal.detach().cpu().numpy().reshape(-1)  

    for _ in range(n_iter):
        grad = np.zeros_like(refined)   
        for i, (x, y) in enumerate(refined):
            print("waypoint {}: ({:.2f}, {:.2f})".format(i, x, y))
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                d = dist_map[yi, xi]
                print("  distance to obstacle: {:.4f}".format(d))
                if d < safe_dist:
                    if d > 0:
                        # move to the closest free space
                        # print("gradient_map:",np.gradient(dist_map))
                        grad_obs = alpha * (safe_dist - d) * np.array(np.gradient(dist_map))[::-1, yi, xi]
                        print("  grad obs: ({:.4f}, {:.4f})".format(grad_obs[0], grad_obs[1]))
                        grad[i] += grad_obs
                    elif d == 0:
                        print("  on obstacle, large grad obs")
                        grad_obs = alpha * safe_dist * np.array(np.gradient(dist_map))[::-1, yi, xi]
                        grad[i] += grad_obs
                        print("  grad obs: ({:.4f}, {:.4f})".format(grad_obs[0], grad_obs[1]))
            if 0 < i < len(refined) - 1:
                grad_smooth = gamma * (refined[i - 1] - 2 * refined[i] + refined[i + 1])
                grad[i] += grad_smooth
                print("grad smooth: ({:.4f}, {:.4f})".format(grad_smooth[0], grad_smooth[1]))
        refined += grad
    refined_world = []
    for (px, py) in refined:
        world_x = py * 0.1
        world_y = (px - w // 2) * 0.1
        refined_world.append([world_x, world_y])
    return np.array(refined_world)

def visualize_pointcloud(points, num_points=200000):
    """
    points: (N, 3) Tensor 
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if points.shape[0] > num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c=points[:, 2], cmap='jet')

    ax.set_xlabel('X(right)')
    ax.set_ylabel('Y(down)')
    ax.set_zlabel('Z(forward)')
    ax.set_title('3D Point Cloud from Depth')
    ax.view_init(elev=-90, azim=-90)
    plt.show()  
    plt.savefig("./pointcloud.png")
    plt.close(fig)

def to_global_coords(positions, curr_pos, curr_yaw):
    """
    Convert positions to global coordinates

    Args:
        positions (list): positions to convert
        curr_pos (list): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in global coordinates
    """
    positions = np.array(positions)
    curr_pos = np.array(curr_pos)
    curr_yaw = float(curr_yaw)
    rotmat = yaw_rotmat(curr_yaw).T
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return positions.dot(rotmat) + curr_pos

def get_pose_from_position(predicted_position: np.array):
    """
    Get yaw from position and then return the pose

    Args:
        predicted_position (np.array): positions
    """
    delta = predicted_position[1:] - predicted_position[:-1]
    yaws = np.arctan2(delta[:, 1], delta[:, 0])
    quaternions = np.array([p.getQuaternionFromEuler([0, 0, yaw]) for yaw in yaws])
    poses = np.concatenate([predicted_position[1:], quaternions], axis=1)
    return poses,yaws

def draw(history_pose: np.ndarray,
         predicted_pose: np.ndarray,
         point_goal: np.ndarray = None,
         rgb: np.ndarray = None,
         save_path: str = None):
    """
    history_pose: (H, 2)
    predicted_pose: (P, 2)
    point_goal: (2,)
    rgb: (H, W, 3)
    save_path: str
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  


    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Observation")
    else:
        axes[0].text(0.5, 0.5, "No RGB Provided", ha='center', va='center', fontsize=12)
    axes[0].axis('off')

    ax = axes[1]
    if history_pose is not None:
        ax.plot(history_pose[:, 0], history_pose[:, 1], 'bo-', label='History Pose')
    if predicted_pose is not None:
        ax.plot(predicted_pose[:, 0], predicted_pose[:, 1], 'ro--', label='Predicted Pose')
    if point_goal is not None:
        ax.plot(point_goal[0], point_goal[1], 'g*', markersize=15, label='Point Goal')

    ax.legend(loc='best')
    ax.set_title("Trajectory")
    ax.axis('equal')
    ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def draw_occu(traj,occupancy):
    plt.figure(figsize=(8, 8))
    occupancy = occupancy.squeeze().cpu().numpy()
    # print("occupancy.shape:",occupancy.shape)   
    plt.imshow(occupancy, cmap='gray', origin='lower')
    traj_pixels = []
    # H_o, W_o = occupancy.shape
    H_o = 50
    W_o = 20
    print("H_o, W_o:",H_o, W_o)
    for waypoint in traj:
        pixel_x = int((-waypoint[1] * 10) + H_o // 2)
        pixel_y = int(waypoint[0] * 10)
        traj_pixels.append((pixel_x, pixel_y))
        # print("pixel_x:",pixel_x)
        # print("pixel_y:",pixel_y)
    traj_pixels = np.array(traj_pixels)
    plt.plot(traj_pixels[:, 0], traj_pixels[:, 1], 'ro--', label='Predicted Traj')
    plt.title('Occupancy Map with Trajectory')
    plt.legend()
    plt.show()


def distance_of_pixel_to_obstacle(occupancy, pixel_x, pixel_y):
    """
    Compute distance to obstacle from a pixel
    Args:
        occupancy (torch.Tensor): (H_o, W_o), 1 for obstacle, 0 for free space, 2 for unknown.
        pixel_x (int): x coordinate of the pixel
        pixel_y (int): y coordinate of the pixel
    Returns:
        distance (float): distance to obstacle from the pixel
    """
    occupancy = occupancy.squeeze().cpu().numpy()
    obstacle = np.argwhere(occupancy == 0)
    target = np.array([[pixel_y, pixel_x]]) 
    if obstacle.shape[0] == 0:
        return 100.0
    min_distance = np.min(distance.cdist(target, obstacle))
    return min_distance

def distance_to_obstacle(occupancy, traj):
    """
    Compute distance to obstacle along the trajectory
    Args:
        occupancy (torch.Tensor): (H_o, W_o), 1 for obstacle, 0 for free space, 2 for unknown.
        traj (list): list of actions
    Returns:
        distance (float): distance to obstacle along the trajectory
    """
    H_o, W_o = occupancy.shape
    x, y = W_o // 2, H_o - 1  # start from bottom center
    distance = 0.0
    for waypoint in traj:
        pixel_x = int((waypoint[0] / 0.1) + W_o // 2)
        pixel_y = int(H_o - 1 - (waypoint[1] / 0.1))    
        distance += distance_of_pixel_to_obstacle(occupancy, pixel_x, pixel_y)
    return distance 
        
def choose_safer_traj(traj_preds, depth, goal, gt_position=None):
    start_time = time.time()
    occupancy = depth_to_occupancy(depth, intrics, grid_size=0.1)  # (B, H_o, W_o)
    occ_time = time.time() 
    # scores = []
    # for traj in traj_preds:
    #     score = distance_to_obstacle(occupancy[0], traj)
    #     scores.append(score)
    compute_time = time.time()
    best_idx = 0
    print("bset _idx:", best_idx)
    if gt_position is not None:
        draw_occu(gt_position, occupancy[0])
    draw_occu(traj_preds[best_idx], occupancy[0])
    # refined_traj = refine_path(occupancy[0].cpu().numpy(), np.array(traj_preds[best_idx]), goal, n_iter=10)
    # draw_occu(refined_traj, occupancy[0])
    print("Occupancy time: {:.4f}s, Compute time: {:.4f}s".format(occ_time - start_time, compute_time - occ_time))
    return traj_preds[best_idx]

def depth_to_occupancy(depth, intrics, grid_size=0.1):
    """
    Convert depth image to local occupancy map.
    Args:
        depth (torch.Tensor): Depth image (B, 1, H, W), depth values in the range 0~255 (corresponding to 0~10 meters).
        config: Camera intrinsics with fx, fy, cx, cy.
        grid_size (float): The size (in meters) of each grid cell (0.1 meter for 10cm).
    Returns:
        occupancy (torch.Tensor): (B, H_o, W_o), 1 for obstacle, 0 for free space, 2 for unknown.
    """
    start_time = time.time()    
    B, _, H, W = depth.shape
    # print("Depth[0,0,125,:]:", depth[0,0,:,125])
    device = depth.device

    #  Create pixel coordinate grids
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device),
        indexing='ij'
    )
    x = x.reshape(-1)
    y = y.reshape(-1)

    #  Convert depth to meters
    z = (depth.reshape(B, -1) / 255.0) * 10.0  # (B, H*W)
    # print("Depth range: [{:.2f}, {:.2f}] meters".format(z.min().item(), z.max().item()))

    fx, fy, cx, cy = intrics.camera_fx, intrics.camera_fy, intrics.camera_cx, intrics.camera_cy
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    points = torch.stack([X, Y, Z], dim=-1)  # (B, H*W, 3)
    # print("Points[0,125:130,:]:", points[0,125:135, :])
    # visualize_pointcloud(points[0])
    points_time = time.time()   
    occupancy_maps = []

    for b in range(B):
        pts = points[b]
        # visualize_pointcloud(pts)
        #  Filter out points where y < -0.2 or z > 2.0
        valid_mask = (pts[:, 1] >= -0.2) & (pts[:, 2] <= 2.0)
        pts = pts[valid_mask]
        # visualize_pointcloud(pts)
        if pts.shape[0] == 0:
            occupancy_maps.append(torch.full((1, 1), 2, device=device))  # No valid points, return empty map
            continue
        #  Define the boundaries based on filtered points
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        # z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        z_min, z_max = torch.tensor(0.0, device=device), pts[:, 2].max()
        # print("Point cloud range x: [{:.2f}, {:.2f}], z: [{:.2f}, {:.2f}]".format(x_min.item(), x_max.item(), z_min.item(), z_max.item()))
        #  Compute grid size based on point cloud range
        # W_o = int(torch.ceil((x_max - x_min) / grid_size).item()) + 1
        # W_o = int(torch.ceil(2 * max(torch.abs(x_min), torch.abs(x_max)) / grid_size).item()) + 1
        # H_o = int(torch.ceil((z_max - z_min) / grid_size).item()) + 1
        W_o = 50
        H_o = 20
        occupancy = torch.full((H_o, W_o), 1.0, device=device)  # Initially set all cells to unknown 

        occupancy_time = time.time()
        #  Map points to the grid
        # gx = ((pts[:, 0] - x_min) / grid_size).long()
        # gx = ((pts[:, 0] - pts[:, 0].min()) / grid_size).long() if pts[:, 0].max().abs() < pts[:, 0].min().abs() else ((pts[:, 0] - pts[:, 0].min() - (pts[:, 0].max() + pts[:, 0].min())) / grid_size).long()
        # gx = ((pts[:, 0] - pts[:, 0].min()) / grid_size).long()
        # x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2
        gx = (pts[:, 0] / grid_size).long() + W_o // 2
        # print("gx range: [{}, {}]".format(gx.min().item(), gx.max().item()))
        gz = ((pts[:, 2] - z_min) / grid_size).long()

        # Clamp the indices to avoid out-of-bounds access
        # gx = torch.clamp(gx, 0, W_o - 1)
        # gz = torch.clamp(gz, 0, H_o - 1)

        #  Mark obstacles (1) and free space (0)
        # for i in range(pts.shape[0]):
        #     if pts[i, 1] < 0.6:  # Mark obstacle if y < 0.8
        #         occupancy[gz[i], gx[i]] = 0.0
        #     else:  # Otherwise mark as free space
        #         if occupancy[gz[i], gx[i]] != 0.0:
        #             occupancy[gz[i], gx[i]] = 1.0
        is_obstacle = pts[:, 1] < 0.6
        is_free = ~is_obstacle
        occupancy[gz[is_obstacle], gx[is_obstacle]] = 0.0
        # mask_free = occupancy[gz[is_free], gx[is_free]] != 0.0
        # occupancy[gz[is_free][mask_free], gx[is_free][mask_free]] = 1.0
        mark_time = time.time() 
        occupancy_maps.append(occupancy)
        dt = euclidean_distance_transform(occupancy)
        print("dt:",dt) 
        plt.figure(figsize=(8, 8))
        plt.imshow(dt.cpu().numpy(), cmap='jet', origin='lower')
        plt.title('Distance Transform')
        plt.colorbar(label='Distance to Nearest Obstacle (m)')
        plt.show()
    print("Points time: {:.4f}s, Occupancy time: {:.4f}s, Mark time: {:.4f}s".format(points_time - start_time, occupancy_time - points_time, mark_time - occupancy_time))
    return torch.stack(occupancy_maps, dim=0)  # (B, H_o, W_o)


def updata_state(s: Simulator, robot: ManipulationRobot, history_rgb: torch.Tensor, history_depth: torch.Tensor, history_pose: torch.Tensor, history_global_position: np.array):
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
        history_global_position = np.concatenate([history_global_position[1:, :], robot.get_position()[:2].reshape(1, 2)],axis=0)
        history_pose = to_local_coords(
            history_global_position.tolist(),
            robot.get_position()[:2].tolist(),
            robot.get_rpy()[2]
        )
        history_pose = torch.from_numpy(history_pose).unsqueeze(0).to("cuda:0").float()
    else:
        history_rgb = torch.cat([history_rgb, cur_rgb.unsqueeze(1)], dim=1)
        history_depth = torch.cat([history_depth, cur_depth.unsqueeze(1)], dim=1)
        history_global_position = np.concatenate([history_global_position[1:, :], robot.get_position()[:2].reshape(1, 2)],axis=0) 
        history_pose = to_local_coords(
            history_global_position.tolist(),   
            robot.get_position()[:2].tolist(),
            robot.get_rpy()[2]
        )
        history_pose = torch.from_numpy(history_pose).unsqueeze(0).to("cuda:0").float() 
    # print("updated history position: {}".format(history_pose))
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
        if np.abs(yaw_diff) < (15 * np.pi / 180):
            actions.append(0)
        # left
        elif yaw_diff >= (15 * np.pi / 180):
            turns = int(yaw_diff / (15 * np.pi / 180)) + 1
            for _ in range(turns):
                actions.append(1)
            actions.append(0)
        # right
        else:
            turns = int(-yaw_diff / (15 * np.pi / 180)) + 1
            for _ in range(turns):
                actions.append(2)
            actions.append(0)
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
