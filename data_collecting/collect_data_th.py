import logging
import os
import argparse
import shutil
from sys import platform
import glob
from typing import Tuple, List
import math
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import cv2 as cv
import time

import numpy as np
from PIL import Image, ImageDraw
import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator

ENABLE_TIME = True
now = time.perf_counter

# ------------------------------- 强健的删除 -------------------------------

def safe_remove_path(path: str):
    """
    强制删除文件/目录/符号链接（即使是坏链接）。
    """
    if not os.path.lexists(path):
        return
    try:
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        pass

# ------------------------------- 工具函数 -------------------------------

def load_floor_count(scene_path: str) -> int:
    """读取楼层数量"""
    floors_file = os.path.join(scene_path, "floors.txt")
    if not os.path.exists(floors_file):
        return 1
    with open(floors_file, 'r') as f:
        return len([line for line in f.readlines() if line.strip() != ""])

def load_floor_heights(scene_path: str) -> List[float]:
    """读取每层的高度值"""
    floors_file = os.path.join(scene_path, "floors.txt")
    if not os.path.exists(floors_file):
        return []
    vals: List[float] = []
    with open(floors_file, 'r') as f:
        for line in f:
            s = line.strip()
            if s != "":
                vals.append(float(s))
    return vals

def load_transform_params(scene_path: str, floor_id: int) -> Tuple[float, float, float]:
    """
    读取第 floor_id 层的变换参数
    返回 (scale, offset_x, offset_y)
    """
    scale_file = os.path.join(scene_path, "scale.txt")
    offset_file = os.path.join(scene_path, "offset.txt")
    
    # 读取 scale
    scale = 1.0
    if os.path.exists(scale_file):
        with open(scale_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != ""]
            if floor_id < len(lines):
                scale = float(lines[floor_id])
    
    # 读取 offset
    offset_x, offset_y = 0.0, 0.0
    if os.path.exists(offset_file):
        with open(offset_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != ""]
            if floor_id < len(lines):
                parts = lines[floor_id].split()
                if len(parts) >= 2:
                    offset_x = float(parts[0])
                    offset_y = float(parts[1])
    
    return scale, offset_x, offset_y

def load_scene_objects(scene_path: str, scene_name: str):
    """
    加载场景的 JSON 文件，如果存在同名 JSON 则返回对象列表
    
    Args:
        scene_path: 场景目录路径
        scene_name: 场景名称
    
    Returns:
        objects_by_floor: 字典 {floor_id: [object_list]}，如果没有 JSON 返回 None
    """
    json_file = os.path.join(scene_path, f"{scene_name}.json")
    print("json_file: ", json_file)
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_file):
        #print("not exist")
        return None
    
    try:
        with open(json_file, 'r') as f:
            objects_list = json.load(f)  # 直接是对象列表
        
        # 按楼层组织对象
        objects_by_floor = {}
        
        for obj in objects_list:
            # 检查必需字段
            if "floor" not in obj:
                continue
            if "agent_position" not in obj or len(obj["agent_position"]) < 2:
                continue
            
            # 获取楼层信息（JSON 中 floor 是 1-based）
            floor = obj["floor"]
            #floor_id = floor - 1  # 转换为 0-based 索引
            floor_id = floor
            # 按楼层分组
            if floor_id not in objects_by_floor:
                objects_by_floor[floor_id] = []
            
            objects_by_floor[floor_id].append(obj)
        
        if objects_by_floor:
            print(f"    ✓ Loaded object JSON: {os.path.basename(json_file)}")
            for floor_id in sorted(objects_by_floor.keys()):
                print(f"      Floor {floor_id}: {len(objects_by_floor[floor_id])} objects")
                pass
        else:
            print(f"    ✗ No valid objects found in JSON")
            return None
        
        return objects_by_floor
    
    except Exception as e:
        print(f"    ✗ Error loading JSON {json_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_object_info(traj_dir: str, 
                     obj: dict, 
                     scale: float, 
                     offset_x: float, 
                     offset_y: float):
    """
    保存轨迹对应的 object 信息到 object/object.json
    
    Args:
        traj_dir: 轨迹目录
        obj: 对象信息字典（来自原始 JSON）
        scale: 坐标变换 scale 参数
        offset_x: 坐标变换 x 偏移
        offset_y: 坐标变换 y 偏移
    """
    object_dir = os.path.join(traj_dir, "object")
    os.makedirs(object_dir, exist_ok=True)
    
    # 提取对象信息（字段名与原始 JSON 一致）
    object_category = obj.get("object_category", "unknown")
    object_id = obj.get("object_id", "unknown")
    position = obj.get("agent_position", [0, 0, 0])
    rotation = obj.get("agent_rotation")
    
    # 计算在 floorplan 上的像素坐标
    if len(position) >= 2:
        px, py = world_to_floorplan(position[0], position[1], scale, offset_x, offset_y)
        floorplan_pos = [px, py]
    else:
        floorplan_pos = [0, 0]
    
    # 构建输出 JSON（保持与原始 JSON 相同的字段名）
    object_info = {
        "object_category": object_category,
        "object_id": object_id,
        "agent_position": position,
        "agent_rotation": rotation,
        "floorplan_pos": floorplan_pos
    }
    
    # 保存 JSON
    json_path = os.path.join(object_dir, "object.json")
    with open(json_path, 'w') as f:
        json.dump(object_info, f, indent=2)

def _process_and_save_single_frame(args):
    """
    处理并保存单帧的 RGB 和 Depth（顶层函数，用于多进程）
    
    Args:
        args: (frame_index, rgb_array, depth_array, rgb_dir, depth_dir)
    """
    import numpy as np
    from PIL import Image
    import os
    
    p, rgb_array, depth_array, rgb_dir, depth_dir= args
    
    # 处理并保存 RGB
    rgb_uint8 = (255 * rgb_array[:, :, :3]).astype(np.uint8)
    # print suppressed
    Image.fromarray(rgb_uint8).save(os.path.join(rgb_dir, f"rgb_{p}.png"))
    
    # 处理并保存 Depth
    depth_m = np.linalg.norm(depth_array[:, :, :3], axis=2)
    depth_m = np.clip(depth_m + 1e-8, None, 10) * 25.5
    depth_uint8 = depth_m.astype(np.uint8)
    Image.fromarray(depth_uint8).save(os.path.join(depth_dir, f"depth_{p}.png"))
    
    return p  # 返回索引，用于进度跟踪

def copy_floorplan(scene_path: str, scene_name: str, dst_dir: str, floor_id: int):
    """
    复制指定楼层的平面图到输出目录
    查找 denoise_after_dilate_{floor_id}.png 并复制为 floorplan.png
    """
    src = os.path.join(scene_path, f"denoise_after_dilate_{floor_id}.png")
    dst = os.path.join(dst_dir, "floorplan.png")
    # print(src)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        # print(f"    Copied floorplan for floor {floor_id}: {os.path.basename(src)}")
    else:
        # print(f"    Warning: Floorplan not found: {src}")
        pass

def sample_valid_trajectory(scene: StaticIndoorScene, floor_id: int, min_distance=5.0, max_attempts=200):
    """
    在指定楼层随机采样一条满足最小长度要求的有效轨迹，并在相邻点之间插值
    
    插值规则：
    - 对于原始轨迹中的相邻两点 (x, y) 和 (x', y')
    - 在它们之间插入中点 ((x+x')/2, (y+y')/2)
    - 原始轨迹有 N 个点，插值后有 2N-1 个点
    
    Args:
        scene: StaticIndoorScene 对象
        floor_id: 楼层索引
        min_distance: 最小轨迹长度（米）
        max_attempts: 最大尝试次数
    
    Returns:
        interpolated_path: 插值后的路径，numpy array of shape (M, 2)，M = 2N-1
        geo: 原始路径的测地距离（插值前的距离）
    """
    for _ in range(max_attempts):
        try:
            # 1. 随机采样起点和终点
            rp1 = scene.get_random_point(floor_id)
            rp2 = scene.get_random_point(floor_id)
            
            # 2. 检查采样是否成功
            if rp1 is None or rp2 is None:
                continue
            
            # 3. 提取坐标（取前两维 x, y）
            p1 = rp1[1][:2]
            p2 = rp2[1][:2]
            
            # 4. 使用 iGibson 的 A* 算法计算最短路径
            path, geo, _ = scene.get_shortest_path(floor_id, p1, p2, entire_path=True)
            
            # 5. 检查路径是否有效且满足最小长度要求
            if path is not None and geo is not None and geo >= min_distance:
                # # 6. 将路径转换为 numpy 数组
                # path_array = np.array(path, dtype=np.float32)  # shape: (N, 2)
                
                # # 7. 开始插值操作
                # N = len(path_array)  # 原始路径点数
                
                # # 8. 特殊情况：如果只有一个点，无法插值，直接返回
                # if N == 1:
                #     return path_array, float(geo)
                
                # # 9. 创建新的轨迹列表，用于存储插值后的点
                # interpolated_points = []
                
                # # 10. 遍历原始轨迹的相邻点对
                # for i in range(N):
                #     # 11. 添加当前原始点
                #     interpolated_points.append(path_array[i])
                    
                #     # 12. 如果不是最后一个点，在当前点和下一个点之间插值
                #     if i < N - 1:
                #         # 13. 获取当前点和下一个点的坐标
                #         x, y = path_array[i]       # 第 i 个点
                #         x_next, y_next = path_array[i + 1]  # 第 i+1 个点
                        
                #         # 14. 计算中点坐标
                #         x_mid = (x + x_next) / 2.0
                #         y_mid = (y + y_next) / 2.0
                        
                #         # 15. 将中点添加到插值点列表
                #         interpolated_points.append(np.array([x_mid, y_mid], dtype=np.float32))
                
                # # 16. 将列表转换为 numpy 数组
                # interpolated_path = np.array(interpolated_points, dtype=np.float32)
                
                # # 17. 验证插值结果的长度
                # expected_length = 2 * N - 1
                # assert len(interpolated_path) == expected_length, \
                #     f"插值错误: 期望 {expected_length} 个点，实际 {len(interpolated_path)} 个点"
                
                # # 18. 返回插值后的路径和原始测地距离
                # return interpolated_path, float(geo)
                return np.array(path, dtype=np.float32), float(geo)
                
        except Exception:
            # 19. 如果出现任何异常，继续尝试下一次采样
            continue
    
    # 20. 如果所有尝试都失败，返回 None
    return None, None

def sample_object_goal_trajectory(scene: StaticIndoorScene, 
                                   floor_id: int, 
                                   goal_position: List[float],
                                   goal_direction: List[float],
                                   min_distance: float = 5.0, 
                                   max_attempts: int = 200):
    """
    采样一条以指定物体位置为终点的轨迹
    """
    # 提取目标点的 x, y 坐标
    goal_xy = np.array([goal_position[0], goal_position[1]], dtype=np.float32)
    
    for _ in range(max_attempts):
        try:
            # 随机采样起点
            rp1 = scene.get_random_point(floor_id)
            
            # 检查采样是否成功
            if rp1 is None:
                continue
            
            # 提取起点坐标
            p1 = rp1[1][:2]
            
            # 使用固定的终点（物体位置）
            p2 = goal_xy
            
            if ENABLE_TIME:
                scene_t = now()
            
            # 使用 iGibson 的 A* 算法计算最短路径
            path, geo, point_del = scene.get_shortest_path(floor_id, p1, p2, entire_path=True)
            if point_del is not None:
                return None, None, point_del
            
            if ENABLE_TIME:
                # print(f"[TIME] SCENE_total(collect_a_single_trajectory_and_maybe_invalid) = {now() - scene_t:.3f}s")
                pass
            
            # 检查路径是否有效且满足最小长度要求
            if path is not None and geo is not None and geo >= min_distance:
                # 将路径转换为 numpy 数组
                path_array = np.array(path, dtype=np.float32)
                # N = len(path_array)
                
                # # 特殊情况：只有一个点
                # if N == 1:
                #     return path_array, float(geo), None
                
                # # 插值操作（与原函数相同）
                # interpolated_points = []
                # for i in range(N):
                #     interpolated_points.append(path_array[i])
                #     if i < N - 1:
                #         x, y = path_array[i]
                #         x_next, y_next = path_array[i + 1]
                #         x_mid = (x + x_next) / 2.0
                #         y_mid = (y + y_next) / 2.0
                #         interpolated_points.append(np.array([x_mid, y_mid], dtype=np.float32))
                
                # interpolated_path = np.array(interpolated_points, dtype=np.float32)
                
                # # 验证插值结果
                # expected_length = 2 * N - 1
                # assert len(interpolated_path) == expected_length, \
                #     f"插值错误: 期望 {expected_length} 个点，实际 {len(interpolated_path)} 个点"

                # return interpolated_path, float(geo), None
                return path_array, float(geo), None
                
        except Exception:
            continue
    
    return None, None, None

def path_directions(path_xy: np.ndarray, goal_directions=None) -> List:
    """计算路径上每个点的前进方向向量"""
    n = len(path_xy)

    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

    def _rotate(v, deg):
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return R @ v

    first_vec = path_xy[1].astype(np.float32) - path_xy[0].astype(np.float32)
    first_dir = _norm(first_vec)
    out_pts  = [path_xy[0].astype(np.float32)]
    out_dirs = [first_dir.copy()]

    prev_pt  = path_xy[0].astype(np.float32)
    prev_dir = first_dir.copy()

    for i in range(1, n):
        cur_pt  = path_xy[i].astype(np.float32)
        seg_vec = cur_pt - prev_pt
        seg_len = np.linalg.norm(seg_vec)
        seg_dir = _norm(seg_vec) if seg_len > 1e-8 else prev_dir.copy()

        # 角度与旋转符号
        dot = float(np.clip(np.dot(prev_dir, seg_dir), -1.0, 1.0))
        deg = float(np.degrees(np.arccos(dot)))
        cross = prev_dir[0] * seg_dir[1] - prev_dir[1] * seg_dir[0]
        sign = 1.0 if cross > 0 else (-1.0 if cross < 0 else 0.0)

        # 计算 steps
        if deg < 15.0 - 1e-8:
            steps = 1  # 至少插值一个点
            k = steps
            # 位置：k+1 等分点，k=1 → t=1/2
            t = 1.0 / (k + 1)
            ipt = (1.0 - t) * prev_pt + t * cur_pt
            # 方向：直接设为该段目标方向（不旋转）
            idir = seg_dir.copy()

            out_pts.append(ipt.astype(np.float32))
            out_dirs.append(idir.astype(np.float32))
        else:
            k = int(deg // 15.0)  # floor
            k = max(1, k)         # 至少 1
            # 方向从 prev_dir 出发，逐点累计旋转 15°
            dir_running = prev_dir.copy()
            for j in range(1, k + 1):
                # 位置：k+1 等分
                t = j / float(k + 1)
                ipt = (1.0 - t) * prev_pt + t * cur_pt
                # 方向：在上一个方向基础上转 15°
                if sign != 0.0:
                    dir_running = _norm(_rotate(dir_running, sign * 15.0))
                # sign == 0（共线）时无需旋转，保持 prev_dir
                out_pts.append(ipt.astype(np.float32))
                out_dirs.append(dir_running.astype(np.float32))

        # 段终点（原始点 i）
        out_pts.append(cur_pt.astype(np.float32))
        out_dirs.append(seg_dir.astype(np.float32))

        prev_pt  = cur_pt
        prev_dir = seg_dir

    """check whether the last direction is right"""
    if goal_directions is not None:
        # 取目标朝向的前两维（忽略z方向）
        goal_dir_2d = np.array(goal_directions[:2], dtype=np.float32)
        goal_dir_norm = np.linalg.norm(goal_dir_2d)
        
        if goal_dir_norm > 1e-8:
            goal_dir_2d = goal_dir_2d / goal_dir_norm
            
            # 获取到达终点时的朝向（倒数第二个点指向倒数第一个点）
            if n >= 2:
                current_dir = out_dirs[-1]  # 最后一个方向向量
            else:
                current_dir = np.array([1.0, 0.0], dtype=np.float32)
            
            # 计算两个向量之间的夹角（无符号）
            def angle_between_vectors(v1, v2):
                """计算两个向量之间的夹角（返回角度制，范围0-180度）"""
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                return angle_deg
            
            # 旋转向量的辅助函数
            def rotate_vector(v, angle_deg):
                """将向量v旋转angle_deg度（正数为逆时针）"""
                angle_rad = np.radians(angle_deg)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                rotation_matrix = np.array([[cos_a, -sin_a], 
                                           [sin_a, cos_a]], dtype=np.float32)
                return rotation_matrix @ v
            
            # 计算初始夹角
            angle_diff = angle_between_vectors(current_dir, goal_dir_2d)
            
            # 如果夹角大于5度，需要插入旋转步骤
            if angle_diff > 5.0:
                # 🔥 修改点：使用叉积判断旋转方向（沿小角度旋转）
                # cross > 0: goal在current左侧，需要逆时针旋转
                # cross < 0: goal在current右侧，需要顺时针旋转
                cross = current_dir[0] * goal_dir_2d[1] - current_dir[1] * goal_dir_2d[0]
                
                # 确定旋转方向：每次旋转15度
                rotation_step = 5.0 if cross > 0 else -5.0
                
                # 终点位置（保持不变）
                goal_position = path_xy[-1].copy()
                
                # 存储插值的位置和方向
                interpolated_positions = []
                interpolated_directions = []
                
                # 当前旋转角度
                accumulated_rotation = 0.0
                current_direction = current_dir.copy()
                
                # 循环旋转，直到接近目标朝向
                while True:
                    # 旋转当前朝向
                    accumulated_rotation += rotation_step
                    current_direction = rotate_vector(current_dir, accumulated_rotation)
                    
                    # 添加插值点（位置不变，只改变朝向)
                    interpolated_positions.append(goal_position.copy())
                    interpolated_directions.append(current_direction.copy())
                    
                    # 检查是否接近目标朝向
                    remaining_angle = angle_between_vectors(current_direction, goal_dir_2d)
                    if remaining_angle <= 5.0:
                        break
                    
                    # 防止无限循环（理论上不会发生，但保险起见）
                    if abs(accumulated_rotation) > 360.0:
                        break
                
                # 将原始路径和插值点合并
                if interpolated_positions:
                    out_pts = np.vstack([out_pts, np.array(interpolated_positions)])
                    out_dirs = np.vstack([out_dirs, np.array(interpolated_directions)])
            
            # 确保最后一个方向是目标方向
            out_dirs[-1] = goal_dir_2d
    
    return [out_pts, out_dirs]

def world_to_floorplan(x: float, y: float, scale: float, offset_x: float, offset_y: float) -> Tuple[int, int]:
    """
    将世界坐标 (x, y) 转换为平面图像素坐标
    返回 (pixel_x, pixel_y)
    """
    px = int(x * scale + offset_x)
    py = int(y * scale + offset_y)
    return px, py

def draw_trajectory_on_floorplan(floorplan_path: str, 
                                  traj_xy: np.ndarray, 
                                  scale: float, 
                                  offset_x: float, 
                                  offset_y: float,
                                  output_path: str):
    """
    在平面图上绘制轨迹点
    每隔 1/10 的路径点数（向上取整）标记一个红点
    """
    if not os.path.exists(floorplan_path):
        # print(f"    Warning: Floorplan not found for trajectory visualization: {floorplan_path}")
        return
    
    # 加载平面图
    img = Image.open(floorplan_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 计算采样间隔
    n_points = len(traj_xy)
    interval = math.ceil(n_points / 10)  # 向上取整
    
    # 绘制红点
    radius = 3  # 红点半径
    for i in range(0, n_points, interval):
        x, y = traj_xy[i]
        px, py = world_to_floorplan(x, y, scale, offset_x, offset_y)
        
        # 绘制圆形红点
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], 
                     fill='red', outline='red')
    
    if n_points > 0:
        x_start, y_start = traj_xy[0]  # 起点
        px_start, py_start = world_to_floorplan(x_start, y_start, scale, offset_x, offset_y)
        
        # 绘制蓝色圆点（起点）
        draw.ellipse([px_start - radius, py_start - radius, 
                     px_start + radius, py_start + radius], 
                     fill='blue', outline='blue')
        
    if n_points > 1:
        x_end, y_end = traj_xy[-1]  # 终点
        px_end, py_end = world_to_floorplan(x_end, y_end, scale, offset_x, offset_y)
        
        # 绘制绿色圆点（终点）
        draw.ellipse([px_end - radius, py_end - radius, 
                     px_end + radius, py_end + radius], 
                     fill='green', outline='green')
    
    # 保存图像
    img.save(output_path)

def render_traj(sim: Simulator, 
                traj_xy_dir: np.ndarray, 
                floor_height: float, 
                out_dir: str, 
                traj_id: int,
                scene_path: str,
                scene_name: str,
                floor_id: int):
    """
    沿轨迹渲染 RGB 和深度图像，保存相机位姿，并生成轨迹可视化
    优化版本：先收集所有帧，再并行保存
    """
    os.makedirs(out_dir, exist_ok=True)
    
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    poses = []
    
    # ===== 阶段 1: 收集所有帧数据（不保存图片）=====
    # print(f"    [Phase 1/2] Rendering {len(traj_xy_dir)} frames...")
    
    if ENABLE_TIME:
        render_t = now()
    
    # 存储所有帧的数组
    all_rgb_arrays = []
    all_depth_arrays = []
    
    for p, (x, y, dx, dy) in enumerate(traj_xy_dir):
        # 相机高度：楼层高度 + 0.85米
        z = floor_height + 0.85
        tar_x, tar_y, tar_z = x + dx, y + dy, floor_height + 0.85

        if ENABLE_TIME:
            scene_t = now()
        
        sim.renderer.set_camera([float(x), float(y), float(z)],
                                [float(tar_x), float(tar_y), float(tar_z)],
                                [0.0, 0.0, 1.0])

        if ENABLE_TIME:
            # print(f"  [TIME] Step {p}: Init camera = {now() - scene_t:.3f}s")
            pass
        
        if ENABLE_TIME:
            scene_t = now()
            
        #with Profiler("Render"):
        frames = sim.renderer.render(modes=("rgb", "3d"))
        
        if ENABLE_TIME:
            # print(f"  [TIME] Step {p}: Render = {now() - scene_t:.3f}s")
            pass
        
        # 只存储数组，不保存图片
        rgb_frame = frames[0]
        depth_frame = frames[1]
        
        all_rgb_arrays.append(rgb_frame)
        all_depth_arrays.append(depth_frame)
        
        poses.append([x, y, tar_x, tar_y])
    
    if ENABLE_TIME:
        total_render_time = now() - render_t
        # print(f"[TIME] Phase 1 - Total render time: {total_render_time:.3f}s")
        # print(f"[TIME] Phase 1 - Average per frame: {total_render_time / len(traj_xy_dir):.3f}s")
        pass
    
    # ===== 阶段 2: 并行保存所有图片 =====
    # print(f"    [Phase 2/2] Saving {len(all_rgb_arrays)} images in parallel...")
    
    if ENABLE_TIME:
        save_t = now()
    

    def _save_one(p, rgb_array, depth_array, rgb_dir, depth_dir):
        # --- RGB ---
        rgb_uint8 = (255 * rgb_array[:, :, :3]).astype(np.uint8)
        rgb_uint8_bgr = cv.cvtColor(rgb_uint8, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(rgb_dir, f"rgb_{p}.png"), rgb_uint8_bgr,
           [cv.IMWRITE_PNG_COMPRESSION, 0])
        # --- Depth ---
        depth_m = np.linalg.norm(depth_array[:, :, :3], axis=2)
        depth_uint8 = (np.clip(depth_m + 1e-8, None, 10) * 25.5).astype(np.uint8)
        cv.imwrite(os.path.join(depth_dir, f"depth_{p}.png"), depth_uint8,
           [cv.IMWRITE_PNG_COMPRESSION, 1])
        return p

    num_workers = 16  # 关键：并发数不要太大
    if ENABLE_TIME:
        save_t = now()
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(_save_one, p, all_rgb_arrays[p], all_depth_arrays[p], rgb_dir, depth_dir)
            for p in range(len(all_rgb_arrays))
        ]
        for _ in as_completed(futs):
            pass
    if ENABLE_TIME:
        total_save_time = now() - save_t
        # print(f"[TIME] Phase 2 - Total save time: {total_save_time:.3f}s")
        # print(f"[TIME] Phase 2 - Average per frame: {total_save_time / len(all_rgb_arrays):.3f}s")
        # print(f"[TIME] Phase 2 - Used {num_workers} threads")
        pass

    # ===== 阶段 3: 保存 pose 数据和可视化 =====
    poses = np.asarray(poses, dtype=np.float32)
    np.savetxt(os.path.join(out_dir, f"traj_{traj_id}.txt"), poses, fmt="%.6f", delimiter=" ")
    np.save(os.path.join(out_dir, f"traj_{traj_id}.npy"), poses)
    
    # 生成轨迹可视化图
    scale, offset_x, offset_y = load_transform_params(scene_path, floor_id)
    scene_out_parent = os.path.dirname(out_dir)
    floorplan_path = os.path.join(scene_out_parent, "floorplan.png")
    traj_xy = traj_xy_dir[:, :2]
    traj_vis_path = os.path.join(out_dir, f"traj_{traj_id}.png")
    draw_trajectory_on_floorplan(floorplan_path, traj_xy, scale, offset_x, offset_y, traj_vis_path)

# ------------------------------- 主流程 -------------------------------

def process_scene(dataset_path: str,
                  scene_name: str,
                  output_path: str,
                  num_trajectories: int = 200,
                  min_distance: float = 5.0,
                  headless: bool = True):
    """处理单个场景的完整流程"""
    scene_src = os.path.join(dataset_path, scene_name)
    # print(f"\n{'='*70}\nProcessing scene: {scene_name}\n{'-'*70}")

    print(scene_src , '\n', scene_name)

    # 1) 加载场景元数据
    if ENABLE_TIME:
        scene_t = now()
    
    # print("  [1/4] Load scene metadata ...")
    num_floors = load_floor_count(scene_src)
    floor_heights = load_floor_heights(scene_src)
    # print(f"    Floors : {num_floors}")

    map_src = os.path.join(scene_src, "map.txt")
    map_list = []
    try:
        with open(map_src, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                map_list.append(line)
    except FileNotFoundError:
        print("文件未找到。")

    if floor_heights:
        for i, h in enumerate(floor_heights):
            # print(f"      Floor {i}: {h:.3f} m")
            pass

    if ENABLE_TIME:
        # print(f"[TIME] SCENE_total(load_meta_data) = {now() - scene_t:.3f}s")
        pass
    
    # ===== 新增：加载 Object JSON（如果存在）=====
    # print("  [1.5/4] Check for object goal JSON ...")
    objects_by_floor = load_scene_objects(scene_src, scene_name)
    print(scene_src, '\n', scene_name)
    is_object_goal_scene = objects_by_floor is not None
    
    if is_object_goal_scene:
        print(f"    ✓ Object goal mode enabled")
        for floor_id, objs in objects_by_floor.items():
            print(f"      Floor {floor_id}: {len(objs)} objects")
            pass
    else:
        # print(f"    ○ Standard navigation mode (no object JSON)")
        pass
    # ==========================================

    # 2) 初始化模拟器并加载场景
    print("  [2/4] Init simulator & import scene ...")
    if ENABLE_TIME:
        scene_t = now()
    
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    
    sim = Simulator(
        mode="headless" if headless else "gui_interactive",
        image_width=256,
        image_height=256,
        rendering_settings=settings,
    )
    
    print("intrics:", sim.renderer.get_intrinsics())

    try:
        scene = StaticIndoorScene(scene_name, build_graph=True)
        sim.import_scene(scene)
        print("    ✓ Scene loaded.")
    except Exception as e:
        print(f"    ✗ Error loading scene: {e}")
        sim.disconnect()
        return False

    if ENABLE_TIME:
        # print(f"[TIME] SCENE_total(init_simulator) = {now() - scene_t:.3f}s")
        pass

    # 3) 逐层生成轨迹
    # print("  [3/4] Generate trajectories ...")
    total = 0


    for f in range(num_floors):
        scene_src = os.path.join(dataset_path, scene_name)
        # 场景输出目录名：场景名_楼层号

        floor_out_dir = os.path.join(output_path, f"{scene_name}_{f}")

        # 如果目录已存在，清空内容
        if ENABLE_TIME:
            scene_t = now()
        
        if os.path.exists(floor_out_dir):
            safe_remove_path(floor_out_dir)
        os.makedirs(floor_out_dir, exist_ok=True)

        if ENABLE_TIME:
            pass
        
        # 复制对应楼层的平面图
        copy_floorplan(scene_src, scene_name, floor_out_dir, f)

        h = floor_heights[f] if f < len(floor_heights) else 0.0
        floor_objects = None
        if is_object_goal_scene and f in objects_by_floor:
            floor_objects = objects_by_floor[int(map_list[f])]
        else:
            pass
        
        """complete the object checking mechnism"""
        succeed, attempts = 0, 0
        max_attempts = num_trajectories * 6
        g = scene.floor_graph[f]
        if floor_objects is not None:
            for obj in floor_objects:
                x = obj["agent_position"][0]
                y = obj["agent_position"][1]
                target_world = np.array([x, y], dtype=np.float32)
                target_map = tuple(scene.world_to_map(target_world))
                """if the height of the object is too high, then remove it"""
                if not g.has_node(target_map) or obj["obj_position"][2] > h + 1.6 :
                    print("yes")
                    floor_objects.remove(obj)

        while succeed < num_trajectories and attempts < max_attempts:
            attempts += 1
            goal_direction = None
            # ===== 新增：根据模式选择不同的采样函数 =====
            if floor_objects is not None and len(floor_objects) > 0:
                # Object Goal 模式：随机选择一个物体作为目标
                """modify the objects in the list in order to make the objects are all in the tarv_maps"""


                target_obj = floor_objects[np.random.randint(0, len(floor_objects))]
                goal_position = target_obj["agent_position"]
                goal_direction = target_obj["agent_rotation"]
                
                if ENABLE_TIME:
                    scene_t = now()
                
                """get the valid trajectory"""
                path_xy, geo, point_del = sample_object_goal_trajectory(
                    scene, f, goal_position, goal_direction, min_distance=min_distance
                )
                
                if point_del is not None:
                    floor_objects = [obj for obj in floor_objects 
                     if not (np.allclose(obj["agent_position"][:2], point_del[:2], atol=1e-6))]

                if ENABLE_TIME:
                    # print(f"[TIME] SCENE_total(collect_a_single_trajectory) = {now() - scene_t:.3f}s")
                    pass
                
            else:
                if ENABLE_TIME:
                    scene_t = now()
                
                # 标准模式：随机起点和终点
                path_xy, geo = sample_valid_trajectory(
                    scene, f, min_distance=min_distance
                )
                
                if ENABLE_TIME:
                    # print(f"[TIME] SCENE_total(collect_a_single_trajectory) = {now() - scene_t:.3f}s")
                    pass
                
                target_obj = None
            # ============================================
            if path_xy is None:
                continue
            

            out = path_directions(path_xy, goal_direction)
            path_xy = out[0]
            dirs = out[1]
            traj = np.concatenate([path_xy, dirs], axis=1)  # [N,4] = [x,y,dx,dy]

            traj_dir = os.path.join(floor_out_dir, f"traj_{succeed}")
            
            # 如果轨迹目录已存在，清空
            if os.path.exists(traj_dir):
                safe_remove_path(traj_dir)

            try:
                if ENABLE_TIME:
                    scene_t = now()
                
                render_traj(sim, traj, h, traj_dir, succeed, scene_src, scene_name, f)
                # ===== 新增：如果是 object goal，保存 object 信息 =====
                if target_obj is not None:
                    scale, offset_x, offset_y = load_transform_params(scene_src, f)
                    save_object_info(traj_dir, target_obj, scale, offset_x, offset_y)
                # ================================================
                succeed += 1
                total += 1
                
                if ENABLE_TIME:
                    # print(f"[TIME] SCENE_total(render_a_trajectory) = {now() - scene_t:.3f}s")
                    pass
                
                if succeed % 10 == 0:
                    # print(f"      Progress: {succeed}/{num_trajectories}")
                    pass
            except Exception as e:
                # print(f"      Warn: failed to render traj {succeed}: {e}")
                pass

        # print(f"      ✓ Floor {f}: generated {succeed} trajs.")
        pass

    sim.disconnect()
    # print(f"\n  ✓ Scene done: total trajectories = {total}")
    return True

# def main():
#     parser = argparse.ArgumentParser(
#         description="iGibson HM3D Data Collection (using your local scenes)",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Example:
#   # Process entire dataset
#   python collect_data.py --dataset_path ./hm3d --output_path ./ --headless
  
#   # Process single scene
#   python collect_data.py --dataset_path ./hm3d/00006-HkseAnWCgqk --output_path ./ --headless
# """)
#     parser.add_argument('--dataset_path', type=str, required=True, 
#                        help='Path to dataset root or single scene directory')
#     parser.add_argument('--output_path', type=str, required=True, 
#                        help='Where to write <dataset>_train')
#     parser.add_argument('--num_trajectories', type=int, default=200, 
#                        help='Trajectories per floor')
#     parser.add_argument('--min_distance', type=float, default=5.0, 
#                        help='Minimum geodesic length (meters)')
#     parser.add_argument('--headless', action='store_true', 
#                        help='Run headless (no GUI)')
#     parser.add_argument('--scene_name', type=str, default=None, 
#                        help='Only process a specific scene name (deprecated, use single scene path instead)')
#     args = parser.parse_args()
#     all_time = time.time()

#     # ===== 判断输入是单场景还是数据集 =====
#     def is_single_scene(path):
#         has_mesh = os.path.exists(os.path.join(path, "mesh_z_up.obj"))
#         has_floors = os.path.exists(os.path.join(path, "floors.txt"))
#         has_trav = any(f.startswith("floor_trav_") and f.endswith(".png") 
#                       for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
#         return has_mesh and has_floors and has_trav
    
#     input_path = os.path.abspath(args.dataset_path)
#     is_single_scene_input = is_single_scene(input_path)

#     if is_single_scene_input:
#         scene_name = os.path.basename(input_path)
#         parent_dir = os.path.dirname(input_path)
#         scenes = [scene_name]
#         dataset_root = parent_dir
#     else:
#         dataset_root = input_path
#         if args.scene_name:
#             scenes = [args.scene_name]
#         else:
#             scenes = [d for d in os.listdir(dataset_root) 
#                       if os.path.isdir(os.path.join(dataset_root, d))]
#             for scene in scenes:
#                 print(scene, '\n')

#     # 创建输出目录
#     out_dir = args.output_path
#     os.makedirs(out_dir, exist_ok=True)

#     # =========================
#     # 检查工具函数
#     # =========================
#     def dir_has_all_trajs(dirpath: str) -> bool:
#         for i in range(150):
#             if not os.path.isdir(os.path.join(dirpath, f"traj_{i}")):
#                 return False
#         return True

#     def dir_has_all_trajs_with_object(dirpath: str) -> bool:
#         for i in range(150):
#             traj_dir = os.path.join(dirpath, f"traj_{i}")
#             if not os.path.isdir(traj_dir):
#                 return False
#             if not os.path.isdir(os.path.join(traj_dir, "object")):
#                 return False
#         return True

#     def is_800_range(scene_name: str) -> bool:
#         try:
#             n = int(scene_name[:5])
#             return 800 <= n <= 899
#         except Exception:
#             return False

#     def scene_has_json(scene_dir: str, scene_name: str) -> bool:
#         return os.path.exists(os.path.join(scene_dir, f"{scene_name}.json"))

#     def must_process_scene(scene_name: str) -> bool:
#         # out_dir 下所有以 "<scene_name>_" 开头的目录
#         try:
#             prefixed_dirs = [
#                 d for d in os.listdir(out_dir)
#                 if d.startswith(scene_name + "_") and os.path.isdir(os.path.join(out_dir, d))
#             ]
#         except FileNotFoundError:
#             prefixed_dirs = []

#         # 没有任何前缀目录 => 必须处理
#         if not prefixed_dirs:
#             return True

#         # 00800..00899 且有 .json => 严格检查（traj 完整 + 每个 traj 有 object/）
#         scene_dir = os.path.join(dataset_root, scene_name)
#         in_800 = is_800_range(scene_name)
#         has_json = scene_has_json(scene_dir, scene_name)

#         if in_800 and has_json:
#             for d in prefixed_dirs:
#                 fullp = os.path.join(out_dir, d)
#                 if not dir_has_all_trajs_with_object(fullp):
#                     return True  # 只要有一个不满足，就重采样
#             return False       # 全满足 => 跳过

#         # 其他情况（不在 008xx，或在 008xx 但无 .json） => 普通检查（仅 traj 完整）
#         for d in prefixed_dirs:
#             fullp = os.path.join(out_dir, d)
#             if not dir_has_all_trajs(fullp):
#                 return True
#         return False

#     # 仅挑出需要处理的场景
#     scenes_to_run = []
#     for scn in scenes:
#         need = must_process_scene(scn)
#         scene_dir = os.path.join(dataset_root, scn)
#         tag = "008xx+json严格" if (is_800_range(scn) and scene_has_json(scene_dir, scn)) else "普通规则"
#         if need:
#             print(f"[DECISION] {scn}: 需要处理（{tag} 检查未通过或不存在前缀目录）")
#             scenes_to_run.append(scn)
#         else:
#             print(f"[DECISION] {scn}: 跳过（{tag} 检查通过）")

#     if not scenes_to_run:
#         print("[SUMMARY] 没有需要处理的场景，程序结束。")
#         return

#     # =========================
#     # 仅对“需要处理”的场景调用 process_scene
#     # =========================
#     ok, fail = 0, 0
#     for i, scn in enumerate(scenes_to_run, 1):
#         try:
#             scene_a_atart = time.time()
#             if process_scene(dataset_root, scn, out_dir,
#                              num_trajectories=args.num_trajectories,
#                              min_distance=args.min_distance,
#                              headless=args.headless):
#                 ok += 1
#                 scene_a_end = time.time()
#                 print("scene_name: ", scn, "_", i, "time cost for sampling this floor: ", scene_a_end - scene_a_atart)
#             else:
#                 fail += 1
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             fail += 1

#     last_time = time.time()
#     print("all time long:  ", last_time - all_time)
#     print(f"[SUMMARY] Processed scenes: {len(scenes_to_run)}, ok={ok}, fail={fail}")

def main():
    parser = argparse.ArgumentParser(
        description="iGibson HM3D Data Collection (using your local scenes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Process entire dataset
  python collect_data.py --dataset_path ./hm3d --output_path ./ --headless
  
  # Process single scene
  python collect_data.py --dataset_path ./hm3d/00006-HkseAnWCgqk --output_path ./ --headless
""")
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to dataset root or single scene directory')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Where to write <dataset>_train')
    parser.add_argument('--num_trajectories', type=int, default=200, 
                        help='Trajectories per floor')
    parser.add_argument('--min_distance', type=float, default=5.0, 
                        help='Minimum geodesic length (meters)')
    parser.add_argument('--headless', action='store_true', 
                        help='Run headless (no GUI)')
    parser.add_argument('--scene_name', type=str, default=None, 
                        help='Only process a specific scene name (deprecated, use single scene path instead)')
    parser.add_argument('--list_path', type=str, default=None,
                        help='Path to list.txt: only process scenes whose names appear in this file')
    args = parser.parse_args()
    all_time = time.time()

    # ===== 判断输入是单场景还是数据集 =====
    def is_single_scene(path):
        has_mesh = os.path.exists(os.path.join(path, "mesh_z_up.obj"))
        has_floors = os.path.exists(os.path.join(path, "floors.txt"))
        has_trav = any(
            f.startswith("floor_trav_") and f.endswith(".png")
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
        return has_mesh and has_floors and has_trav
    
    input_path = os.path.abspath(args.dataset_path)
    is_single_scene_input = is_single_scene(input_path)

    if is_single_scene_input:
        # 单场景模式：忽略 list.txt
        scene_name = os.path.basename(input_path)
        parent_dir = os.path.dirname(input_path)
        scenes = [scene_name]
        dataset_root = parent_dir
    else:
        # 数据集模式
        dataset_root = input_path

        # 如果提供了 list.txt，就读一下
        list_names = None
        if args.list_path is not None:
            list_path = os.path.abspath(args.list_path)
            if os.path.exists(list_path):
                with open(list_path, "r", encoding="utf-8") as f:
                    list_names = {
                        line.strip() for line in f.readlines() if line.strip()
                    }
                print(f"[INFO] Loaded {len(list_names)} scene names from {list_path}")
            else:
                print(f"[WARN] list file not found: {list_path}, ignore --list_path")
                list_names = None

        # 枚举数据集下的所有子目录场景
        all_scenes = [
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ]

        # 优先使用 list.txt 过滤
        if list_names is not None:
            # 只保留同时在数据集目录 & list.txt 中的场景
            scenes = [d for d in all_scenes if d in list_names]
        elif args.scene_name:
            # 没有 list.txt，但指定了单个 scene_name
            if args.scene_name in all_scenes:
                scenes = [args.scene_name]
            else:
                print(f"[WARN] scene_name {args.scene_name} not found under {dataset_root}")
                scenes = []
        else:
            # 没有 list.txt，也没指定 scene_name => 全部场景
            scenes = all_scenes

        for scene in scenes:
            print(scene, '\n')

    # 创建输出目录
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 检查工具函数
    # =========================
    def dir_has_all_trajs(dirpath: str) -> bool:
        for i in range(150):
            if not os.path.isdir(os.path.join(dirpath, f"traj_{i}")):
                return False
        return True

    def dir_has_all_trajs_with_object(dirpath: str) -> bool:
        for i in range(150):
            traj_dir = os.path.join(dirpath, f"traj_{i}")
            if not os.path.isdir(traj_dir):
                return False
            if not os.path.isdir(os.path.join(traj_dir, "object")):
                return False
        return True

    def is_800_range(scene_name: str) -> bool:
        try:
            n = int(scene_name[:5])
            return 800 <= n <= 899
        except Exception:
            return False

    def scene_has_json(scene_dir: str, scene_name: str) -> bool:
        return os.path.exists(os.path.join(scene_dir, f"{scene_name}.json"))

    def must_process_scene(scene_name: str) -> bool:
        # out_dir 下所有以 "<scene_name>_" 开头的目录
        try:
            prefixed_dirs = [
                d for d in os.listdir(out_dir)
                if d.startswith(scene_name + "_") and os.path.isdir(os.path.join(out_dir, d))
            ]
        except FileNotFoundError:
            prefixed_dirs = []

        # 没有任何前缀目录 => 必须处理
        if not prefixed_dirs:
            return True

        # 00800..00899 且有 .json => 严格检查（traj 完整 + 每个 traj 有 object/）
        scene_dir = os.path.join(dataset_root, scene_name)
        in_800 = is_800_range(scene_name)
        has_json = scene_has_json(scene_dir, scene_name)

        if in_800 and has_json:
            for d in prefixed_dirs:
                fullp = os.path.join(out_dir, d)
                if not dir_has_all_trajs_with_object(fullp):
                    return True  # 只要有一个不满足，就重采样
            return False       # 全满足 => 跳过

        # 其他情况（不在 008xx，或在 008xx 但无 .json） => 普通检查（仅 traj 完整）
        for d in prefixed_dirs:
            fullp = os.path.join(out_dir, d)
            if not dir_has_all_trajs(fullp):
                return True
        return False

    # 仅挑出需要处理的场景
    scenes_to_run = []
    for scn in scenes:
        need = must_process_scene(scn)
        scene_dir = os.path.join(dataset_root, scn)
        tag = "008xx+json严格" if (is_800_range(scn) and scene_has_json(scene_dir, scn)) else "普通规则"
        if need:
            print(f"[DECISION] {scn}: 需要处理（{tag} 检查未通过或不存在前缀目录）")
            scenes_to_run.append(scn)
        else:
            print(f"[DECISION] {scn}: 跳过（{tag} 检查通过）")

    if not scenes_to_run:
        print("[SUMMARY] 没有需要处理的场景，程序结束。")
        return

    # =========================
    # 仅对“需要处理”的场景调用 process_scene
    # =========================
    ok, fail = 0, 0
    for i, scn in enumerate(scenes_to_run, 1):
        try:
            scene_a_atart = time.time()
            if process_scene(dataset_root, scn, out_dir,
                             num_trajectories=args.num_trajectories,
                             min_distance=args.min_distance,
                             headless=args.headless):
                ok += 1
                scene_a_end = time.time()
                print("scene_name: ", scn, "_", i, "time cost for sampling this floor: ", scene_a_end - scene_a_atart)
            else:
                fail += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            fail += 1

    last_time = time.time()
    print("all time long:  ", last_time - all_time)
    print(f"[SUMMARY] Processed scenes: {len(scenes_to_run)}, ok={ok}, fail={fail}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
