# import logging
# import os
# import argparse
# import shutil
# from sys import platform
# import glob
# from typing import Tuple, List
# import math
# import json
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from PIL import Image
# import cv2 as cv
# import time
# from typing import List, Dict, Any

# import numpy as np
# from PIL import Image, ImageDraw
# import igibson
# from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
# from igibson.render.profiler import Profiler
# from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
# from igibson.simulator import Simulator

# def load_floor_heights(scene_dir: str) -> List[float]:
#     """
#     加载 scene_dir 下的 floors.txt
#     每一行是一个浮点数，表示一层的层高
#     返回一个列表，索引从 0 开始，对应第 0 层、第 1 层……
#     """
#     floors_path = os.path.join(scene_dir, "floors.txt")
#     heights: List[float] = []

#     if not os.path.exists(floors_path):
#         return heights  # 文件不存在就返回空列表

#     with open(floors_path, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             try:
#                 heights.append(float(s))
#             except ValueError:
#                 # 遇到非法行就跳过
#                 continue
#     return heights


# def load_mapping(scene_dir: str) -> List[int]:
#     """
#     加载 scene_dir 下的 map.txt
#     每一行是一个整数，表示某一层的 floor 映射关系
#     返回一个列表，索引从 0 开始
#     """
#     map_path = os.path.join(scene_dir, "map.txt")
#     floor_map: List[int] = []

#     if not os.path.exists(map_path):
#         return floor_map  # 文件不存在就返回空列表

#     with open(map_path, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             try:
#                 floor_map.append(int(s))
#             except ValueError:
#                 # 遇到非法行就跳过
#                 continue
#     return floor_map

# def clear_json_file(json_path: str):
#     """
#     清空一个 .json 文件的内容，并写入一个空列表 [] 作为有效的 JSON。

#     Args:
#         json_path: 要清空的 json 文件路径
#     """
#     # 如果目录不存在就直接报错更容易发现问题
#     dir_name = os.path.dirname(json_path)
#     if dir_name and not os.path.exists(dir_name):
#         raise FileNotFoundError(f"目录不存在: {dir_name}")

#     # 覆盖写入为空列表，保证依然是合法 JSON
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump([], f, ensure_ascii=False)

# def append_objects_to_json(json_path: str, objects: List[Dict[str, Any]]):
#     """
#     将一个对象列表（每个元素是字典）追加写入到指定的 .json 文件中。
#     文件中的顶层结构是一个 list，新的字典会被 append 到这个 list 的末尾。

#     Args:
#         json_path: 目标 json 文件路径
#         objects:   要追加的对象列表，每个元素都是一个 dict
#     """
#     # 确保目录存在
#     dir_name = os.path.dirname(json_path)
#     if dir_name and not os.path.exists(dir_name):
#         os.makedirs(dir_name, exist_ok=True)

#     # 先读旧内容，如果不存在或格式不对，就当成空列表
#     old_data = []
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, "r", encoding="utf-8") as f:
#                 data = f.read().strip()
#                 if data:  # 非空文件
#                     parsed = json.loads(data)
#                     if isinstance(parsed, list):
#                         old_data = parsed
#         except Exception:
#             # 读失败或格式不对，直接忽略，当成空列表
#             pass

#     # 追加新数据
#     old_data.extend(objects)

#     # 覆盖写回
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(old_data, f, ensure_ascii=False, indent=2)

# def load_scene_objects(scene_dir: str, scene_name: str) -> Dict[int, List[Dict]]:
#     """
#     加载单个场景目录下的 <scene_name>.json，
#     按楼层分组返回对象字典，key 为 0-based 的楼层号，value 为该层的对象列表。

#     返回结构示例：
#         {
#             0: [obj0_floor0, obj1_floor0, ...],   # 第 0 层的所有 object
#             1: [obj0_floor1, obj1_floor1, ...],   # 第 1 层的所有 object
#             ...
#         }

#     Args:
#         scene_dir: 场景所在目录，例如 /path/to/dataset/00000-kfPV7w3FaU5
#         scene_name: 场景名字，例如 00000-kfPV7w3FaU5

#     Returns:
#         objects_by_floor: Dict[int, List[dict]]，如果找不到 json 或解析失败则返回 {}
#     """
#     json_path = os.path.join(scene_dir, f"{scene_name}.json")
#     if not os.path.exists(json_path):
#         logging.warning(f"[load_scene_objects] JSON not found: {json_path}")
#         return {}

#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if not isinstance(data, list):
#             logging.warning(
#                 f"[load_scene_objects] JSON format is not a list: {json_path}"
#             )
#             return {}

#         objects_by_floor: Dict[int, List[Dict]] = {}

#         for obj in data:
#             floor_raw = obj.get("floor", None)
#             if not isinstance(floor_raw, int):
#                 continue

#             # floor 改成 0-based
#             floor_id = floor_raw - 1
#             if floor_id < 0:
#                 continue

#             obj["floor"] = floor_id

#             if floor_id not in objects_by_floor:
#                 objects_by_floor[floor_id] = []
#             objects_by_floor[floor_id].append(obj)

#         return objects_by_floor

#     except Exception as e:
#         logging.error(f"[load_scene_objects] Error loading {json_path}: {e}")
#         return {}



# def process_scene(scene: StaticIndoorScene,
#                   scene_dir: str,
#                   scene_name: str,
#                   args):
#     """
#     对单个场景进行数据采集的核心逻辑。
#     这里先留空，由你自己根据需要实现。

#     Args:
#         scene: 已经构造好的 StaticIndoorScene 对象
#         scene_dir: 场景目录的绝对路径
#         scene_name: 场景名称（目录名）
#         args: main 里解析到的命令行参数
#     """
#     mapping_list = load_mapping(scene_dir)
#     floors_heights = load_floor_heights(scene_dir)
#     objects = load_scene_objects(scene_dir, scene_name)
#     clear_json_file(os.path.join(scene_dir, f"{scene_name}.json"))

#     settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
#     sim = Simulator(
#         mode="headless",
#         image_width=256,
#         image_height=256,
#         rendering_settings=settings,
#     )
#     sim.import_scene(scene)
    
#     print("scene name:", scene_name)

#     """deal with every valid floor"""
#     for fl, f in enumerate(floors_heights):
#         floor_objects = None
#         if mapping_list[fl] in objects:
#             floor_objects = objects[int(mapping_list[fl])]
#         else:
#             pass
    
#         """check with every object in this floor"""
#         if floor_objects is not None:
#             print("not none")
#             g = scene.floor_graph[fl]
#             valid_objects = []
#             for obj in floor_objects:
#                 x = obj["obj_position"][0]
#                 y = obj["obj_position"][1]
#                 z = obj["obj_position"][2]
#                 target_world = np.array([x, y], dtype=np.float32)
#                 target_map = tuple(scene.world_to_map(target_world))
#                 if g.has_node(target_map) or z > f + 1.6:
#                     continue
                
#                 nodes = np.array(g.nodes)
#                 closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
#                 print("shape: ", np.array(closest_node).shape)
#                 print("type: ", type(np.array(closest_node)))
#                 new_target_world = scene.map_to_world(np.array(closest_node))

#                 offset = (new_target_world[0] - target_world[0]) * (new_target_world[0] - target_world[0]) + (new_target_world[1] - target_world[1]) * (new_target_world[1] - target_world[1])
#                 if offset > 0.5:
#                     continue

#                 dir = target_world - new_target_world
#                 norm = math.sqrt(sum(v * v for v in dir))
#                 dir = [v / norm for v in dir]
#                 obj_new = obj.copy()
#                 obj_new["agent_position"] = [new_target_world[0] + -0.36 * dir[0], new_target_world[1] + -0.36 * dir[1], z]
#                 obj_new["agent_rotation"] = dir
#                 obj_new["agent_rotation"].append(0)
#                 valid_objects.append(obj_new)

#             append_objects_to_json(os.path.join(scene_dir, f"{scene_name}.json"), valid_objects)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Object-goal data collection for embodied navigation (iGibson)",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Example:
#   # 单个场景目录
#   python collect_object_goal.py --path ./hm3d/00000-kfPV7w3FaU5

#   # 数据集根目录（下面有很多子场景）
#   python collect_object_goal.py --path ./hm3d

#   # 只处理 list.txt 中列出的这些场景（数据集模式）
#   python collect_object_goal.py --path ./hm3d --scene_list ./list.txt
# """
#     )
#     parser.add_argument(
#         "--path",
#         type=str,
#         required=True,
#         help="场景目录或者数据集根目录",
#     )
#     parser.add_argument(
#         "--scene_list",
#         type=str,
#         default=None,
#         help="当 --path 为数据集目录时，指定一个包含若干场景名（一行一个）的 txt，只处理这些场景",
#     )
#     args = parser.parse_args()

#     input_path = os.path.abspath(args.path)

#     # ---------------- 判定是否是“单个场景目录” ----------------
#     def is_single_scene(path: str) -> bool:
#         """
#         判断 path 是否为单个场景目录：
#         条件：path 下存在 mesh_z_up.obj
#         """
#         return os.path.exists(os.path.join(path, "mesh_z_up.obj"))

#     # 如果给了 scene_list，先读进去
#     scene_filter = None
#     if args.scene_list is not None:
#         list_path = os.path.abspath(args.scene_list)
#         if not os.path.isfile(list_path):
#             raise FileNotFoundError(f"scene_list 文件不存在: {list_path}")
#         with open(list_path, "r", encoding="utf-8") as f:
#             names = [line.strip() for line in f.readlines()]
#         # 去掉空行
#         scene_filter = {n for n in names if n}
#         print(f"[INFO] 使用 scene_list，共 {len(scene_filter)} 个场景名。")

#     if is_single_scene(input_path):
#         # 单场景模式：scene_list 对单场景没意义，直接忽略
#         scene_name = os.path.basename(input_path)
#         scene_dir = input_path
#         scenes = [(scene_name, scene_dir)]
#         print(f"[INFO] Input is a single scene: {scene_name}")
#     else:
#         # 数据集模式：遍历子目录，把其中包含 mesh_z_up.obj 的当作场景
#         dataset_root = input_path
#         scenes = []
#         all_scene_names = set()

#         for d in os.listdir(dataset_root):
#             full = os.path.join(dataset_root, d)
#             if not os.path.isdir(full):
#                 continue
#             # 如果指定了 scene_list，则只保留列表里出现的场景名
#             if scene_filter is not None and d not in scene_filter:
#                 continue
#             if is_single_scene(full):
#                 scenes.append((d, full))
#                 all_scene_names.add(d)

#         print(f"[INFO] Input is a dataset. Selected {len(scenes)} scene(s).")

#         # 如果有 scene_list，提示哪些在 list 里但目录中不存在或不是合法场景
#         if scene_filter is not None:
#             missing = scene_filter.difference(all_scene_names)
#             if missing:
#                 print("[WARN] 下列场景名在数据集目录中没有找到对应合法场景：")
#                 for name in sorted(missing):
#                     print("   ", name)

#     # ---------------- 逐场景构建 StaticIndoorScene 并调用 process_scene ----------------
#     for idx, (scene_name, scene_dir) in enumerate(scenes, 1):
#         print(f"\n[SCENE {idx}/{len(scenes)}] {scene_name}")
#         try:
#             # 构造 iGibson 场景（StaticIndoorScene）
#             scene = StaticIndoorScene(scene_name, build_graph=True)
#             # 调你自己的核心逻辑
#             process_scene(scene, scene_dir, scene_name, args)
#         except Exception as e:
#             import traceback
#             print(f"[ERROR] Failed to process scene {scene_name}: {e}")
#             traceback.print_exc()

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()



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
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw
import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator

def load_floor_heights(scene_dir: str) -> List[float]:
    """
    加载 scene_dir 下的 floors.txt
    每一行是一个浮点数，表示一层的层高
    返回一个列表，索引从 0 开始，对应第 0 层、第 1 层……
    """
    floors_path = os.path.join(scene_dir, "floors.txt")
    heights: List[float] = []

    if not os.path.exists(floors_path):
        return heights  # 文件不存在就返回空列表

    with open(floors_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                heights.append(float(s))
            except ValueError:
                # 遇到非法行就跳过
                continue
    return heights


def load_mapping(scene_dir: str) -> List[int]:
    """
    加载 scene_dir 下的 map.txt
    每一行是一个整数，表示某一层的 floor 映射关系
    返回一个列表，索引从 0 开始
    """
    map_path = os.path.join(scene_dir, "map.txt")
    floor_map: List[int] = []

    if not os.path.exists(map_path):
        return floor_map  # 文件不存在就返回空列表

    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                floor_map.append(int(s))
            except ValueError:
                # 遇到非法行就跳过
                continue
    return floor_map

def clear_json_file(json_path: str):
    """
    清空一个 .json 文件的内容，并写入一个空列表 [] 作为有效的 JSON。

    Args:
        json_path: 要清空的 json 文件路径
    """
    # 如果目录不存在就直接报错更容易发现问题
    dir_name = os.path.dirname(json_path)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"目录不存在: {dir_name}")

    # 覆盖写入为空列表，保证依然是合法 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

def append_objects_to_json(json_path: str, objects: List[Dict[str, Any]]):
    """
    将一个对象列表（每个元素是字典）追加写入到指定的 .json 文件中。
    文件中的顶层结构是一个 list，新的字典会被 append 到这个 list 的末尾。

    Args:
        json_path: 目标 json 文件路径
        objects:   要追加的对象列表，每个元素都是一个 dict
    """
    # 确保目录存在
    dir_name = os.path.dirname(json_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # 先读旧内容，如果不存在或格式不对，就当成空列表
    old_data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if data:  # 非空文件
                    parsed = json.loads(data)
                    if isinstance(parsed, list):
                        old_data = parsed
        except Exception:
            # 读失败或格式不对，直接忽略，当成空列表
            pass

    # 追加新数据
    old_data.extend(objects)

    # 覆盖写回
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=2)

def load_scene_objects(scene_dir: str, scene_name: str) -> Dict[int, List[Dict]]:
    """
    加载单个场景目录下的 <scene_name>.json，
    按楼层分组返回对象字典，key 为 0-based 的楼层号，value 为该层的对象列表。

    返回结构示例：
        {
            0: [obj0_floor0, obj1_floor0, ...],   # 第 0 层的所有 object
            1: [obj0_floor1, obj1_floor1, ...],   # 第 1 层的所有 object
            ...
        }

    Args:
        scene_dir: 场景所在目录，例如 /path/to/dataset/00000-kfPV7w3FaU5
        scene_name: 场景名字，例如 00000-kfPV7w3FaU5

    Returns:
        objects_by_floor: Dict[int, List[dict]]，如果找不到 json 或解析失败则返回 {}
    """
    json_path = os.path.join(scene_dir, f"{scene_name}.json")
    if not os.path.exists(json_path):
        logging.warning(f"[load_scene_objects] JSON not found: {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logging.warning(
                f"[load_scene_objects] JSON format is not a list: {json_path}"
            )
            return {}

        objects_by_floor: Dict[int, List[Dict]] = {}

        for obj in data:
            floor_raw = obj.get("floor", None)
            if not isinstance(floor_raw, int):
                continue

            # floor 改成 0-based
            floor_id = floor_raw - 1
            if floor_id < 0:
                continue

            obj["floor"] = floor_id

            if floor_id not in objects_by_floor:
                objects_by_floor[floor_id] = []
            objects_by_floor[floor_id].append(obj)

        return objects_by_floor

    except Exception as e:
        logging.error(f"[load_scene_objects] Error loading {json_path}: {e}")
        return {}



def process_scene(scene: StaticIndoorScene,
                  scene_dir: str,
                  scene_name: str,
                  args):
    """
    对单个场景进行数据采集的核心逻辑。
    """
    mapping_list = load_mapping(scene_dir)
    floors_heights = load_floor_heights(scene_dir)
    objects = load_scene_objects(scene_dir, scene_name)
    clear_json_file(os.path.join(scene_dir, f"{scene_name}.json"))

    settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
    sim = Simulator(
        mode="headless",
        image_width=256,
        image_height=256,
        rendering_settings=settings,
        device_idx=3
    )
    sim.import_scene(scene)
    
    print("scene name:", scene_name)

    """deal with every valid floor"""
    for fl, f in enumerate(floors_heights):
        floor_objects = None
        if mapping_list[fl] in objects:
            floor_objects = objects[int(mapping_list[fl])]
        else:
            pass
    
        """check with every object in this floor"""
        if floor_objects is not None:
            print("not none")
            g = scene.floor_graph[fl]
            valid_objects = []
            for obj in floor_objects:
                x = obj["obj_position"][0]
                y = obj["obj_position"][1]
                z = obj["obj_position"][2]
                target_world = np.array([x, y], dtype=np.float32)
                target_map = tuple(scene.world_to_map(target_world))
                if g.has_node(target_map) or z > f + 1.6:
                    continue
                
                nodes = np.array(g.nodes)
                closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
                print("shape: ", np.array(closest_node).shape)
                print("type: ", type(np.array(closest_node)))
                new_target_world = scene.map_to_world(np.array(closest_node))

                offset = (new_target_world[0] - target_world[0]) * (new_target_world[0] - target_world[0]) + (new_target_world[1] - target_world[1]) * (new_target_world[1] - target_world[1])
                if offset > 0.5:
                    continue

                dir = target_world - new_target_world
                norm = math.sqrt(sum(v * v for v in dir))
                dir = [v / norm for v in dir]
                obj_new = obj.copy()
                obj_new["agent_position"] = [new_target_world[0] + -0.36 * dir[0], new_target_world[1] + -0.36 * dir[1], z]
                obj_new["agent_rotation"] = dir
                obj_new["agent_rotation"].append(0)
                valid_objects.append(obj_new)

            append_objects_to_json(os.path.join(scene_dir, f"{scene_name}.json"), valid_objects)


def scene_has_agent_rotation(scene_dir: str, scene_name: str) -> bool:
    """
    检查 scene_dir/scene_name.json 中是否存在任意一个对象包含 key 'object_rotation'。
    只要有一个 dict 里有这个 key，就认为已经处理过，返回 True。
    """
    json_path = os.path.join(scene_dir, f"{scene_name}.json")
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False
        for item in data:
            if isinstance(item, dict) and "agent_rotation" in item:
                return True
        return False
    except Exception:
        # 解析失败就按“未处理”看待，返回 False
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Object-goal data collection for embodied navigation (iGibson)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # 单个场景目录
  python collect_object_goal.py --path ./hm3d/00000-kfPV7w3FaU5

  # 数据集根目录（下面有很多子场景）
  python collect_object_goal.py --path ./hm3d

  # 只处理 list.txt 中列出的这些场景（数据集模式）
  python collect_object_goal.py --path ./hm3d --scene_list ./list.txt
"""
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="场景目录或者数据集根目录",
    )
    parser.add_argument(
        "--scene_list",
        type=str,
        default=None,
        help="当 --path 为数据集目录时，指定一个包含若干场景名（一行一个）的 txt，只处理这些场景",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.path)

    # ---------------- 判定是否是“单个场景目录” ----------------
    def is_single_scene(path: str) -> bool:
        """
        判断 path 是否为单个场景目录：
        条件：path 下存在 mesh_z_up.obj
        """
        return os.path.exists(os.path.join(path, "mesh_z_up.obj"))

    # 如果给了 scene_list，先读进去
    scene_filter = None
    if args.scene_list is not None:
        list_path = os.path.abspath(args.scene_list)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"scene_list 文件不存在: {list_path}")
        with open(list_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines()]
        # 去掉空行
        scene_filter = {n for n in names if n}
        print(f"[INFO] 使用 scene_list，共 {len(scene_filter)} 个场景名。")

    if is_single_scene(input_path):
        # 单场景模式：scene_list 对单场景没意义，直接忽略
        scene_name = os.path.basename(input_path)
        scene_dir = input_path
        scenes = [(scene_name, scene_dir)]
        print(f"[INFO] Input is a single scene: {scene_name}")
    else:
        # 数据集模式：遍历子目录，把其中包含 mesh_z_up.obj 的当作场景
        dataset_root = input_path
        scenes = []
        all_scene_names = set()

        for d in os.listdir(dataset_root):
            full = os.path.join(dataset_root, d)
            if not os.path.isdir(full):
                continue
            # 如果指定了 scene_list，则只保留列表里出现的场景名
            if scene_filter is not None and d not in scene_filter:
                continue
            if is_single_scene(full):
                scenes.append((d, full))
                all_scene_names.add(d)

        print(f"[INFO] Input is a dataset. Selected {len(scenes)} scene(s).")

        # 如果有 scene_list，提示哪些在 list 里但目录中不存在或不是合法场景
        if scene_filter is not None:
            missing = scene_filter.difference(all_scene_names)
            if missing:
                print("[WARN] 下列场景名在数据集目录中没有找到对应合法场景：")
                for name in sorted(missing):
                    print("   ", name)

    # -------- 新增：统一根据 object_rotation 过滤要处理的场景 --------
    filtered_scenes = []
    for scene_name, scene_dir in scenes:
        if scene_has_agent_rotation(scene_dir, scene_name):
            print(f"[SKIP] {scene_name}: JSON 中已存在 'object_rotation' 字段，跳过该场景。")
        else:
            filtered_scenes.append((scene_name, scene_dir))

    if not filtered_scenes:
        print("[INFO] 没有需要处理的场景（全部因为含有 'object_rotation' 被跳过）。")
        return

    # ---------------- 逐场景构建 StaticIndoorScene 并调用 process_scene ----------------
    for idx, (scene_name, scene_dir) in enumerate(filtered_scenes, 1):
        print(f"\n[SCENE {idx}/{len(filtered_scenes)}] {scene_name}")
        try:
            # 构造 iGibson 场景（StaticIndoorScene）
            scene = StaticIndoorScene(scene_name, build_graph=True)
            # 调你自己的核心逻辑
            process_scene(scene, scene_dir, scene_name, args)
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process scene {scene_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
