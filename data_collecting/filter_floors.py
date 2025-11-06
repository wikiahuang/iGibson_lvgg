#!/usr/bin/env python3
"""
脚本功能：
1. 自动判断输入是单个场景还是数据集（通过检查mesh_z_up.obj文件）
2. 根据floor_trav_i.png文件来过滤floors.txt中的行
3. 删除不匹配的meshcut_floor_i.png和denoise_after_dilate_i.png文件
4. 重命名保留的文件，使索引从0开始连续
5. 创建map.txt记录原始索引
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set


def is_scene_directory(directory: Path) -> bool:
    """
    判断目录是场景还是数据集

    Args:
        directory: 目录路径

    Returns:
        True表示是场景（包含mesh_z_up.obj），False表示是数据集
    """
    return (directory / 'mesh_z_up.obj').exists()


def extract_floor_indices(directory: Path, pattern: str) -> Set[int]:
    """
    提取指定模式的文件索引

    Args:
        directory: 目录路径
        pattern: 文件名模式，如 'floor_trav_'

    Returns:
        索引集合
    """
    indices = set()
    for file in directory.iterdir():
        if file.is_file() and file.name.startswith(pattern) and file.suffix == '.png':
            # 提取索引部分
            name_without_ext = file.stem
            index_str = name_without_ext[len(pattern):]
            # 确保索引是单个数字
            if index_str.isdigit() and len(index_str) == 1:
                indices.add(int(index_str))
    return indices


def filter_floors_txt(directory: Path, valid_indices: Set[int]) -> None:
    """
    过滤floors.txt文件，只保留有效索引对应的行

    Args:
        directory: 目录路径
        valid_indices: 有效的索引集合
    """
    floors_file = directory / 'floors.txt'
    if not floors_file.exists():
        print(f"  警告: {floors_file} 不存在")
        return

    # 读取所有行
    with open(floors_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤保留有效索引的行
    filtered_lines = []
    for idx in sorted(valid_indices):
        if idx < len(lines):
            filtered_lines.append(lines[idx])

    # 写回文件
    with open(floors_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print(f"  floors.txt: 保留了 {len(filtered_lines)} 行 (索引: {sorted(valid_indices)})")


def remove_invalid_files(directory: Path, valid_indices: Set[int], patterns: List[str]) -> None:
    """
    删除不在有效索引中的文件

    Args:
        directory: 目录路径
        valid_indices: 有效的索引集合
        patterns: 文件名模式列表
    """
    for pattern in patterns:
        for file in list(directory.glob(f'{pattern}*.png')):
            # 提取索引
            name_without_ext = file.stem
            index_str = name_without_ext[len(pattern):]

            if index_str.isdigit() and len(index_str) == 1:
                index = int(index_str)
                if index not in valid_indices:
                    file.unlink()
                    print(f"  删除: {file.name}")


def rename_files(directory: Path, valid_indices: List[int], patterns: List[str]) -> None:
    """
    重命名文件，使索引从0开始连续

    Args:
        directory: 目录路径
        valid_indices: 有效的索引列表（已排序）
        patterns: 文件名模式列表
    """
    # 创建索引映射：旧索引 -> 新索引
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # 为了避免重命名冲突，先将文件重命名为临时名称
    temp_files = []
    for pattern in patterns:
        for old_idx in valid_indices:
            old_file = directory / f'{pattern}{old_idx}.png'
            if old_file.exists():
                temp_file = directory / f'{pattern}temp_{old_idx}.png'
                old_file.rename(temp_file)
                temp_files.append((temp_file, pattern, old_idx))

    # 然后重命名为最终名称
    for temp_file, pattern, old_idx in temp_files:
        new_idx = index_map[old_idx]
        new_file = directory / f'{pattern}{new_idx}.png'
        temp_file.rename(new_file)
        print(f"  重命名: {pattern}{old_idx}.png -> {pattern}{new_idx}.png")


def create_map_file(directory: Path, valid_indices: List[int]) -> None:
    """
    创建map.txt文件，记录保留下来的原始索引

    Args:
        directory: 目录路径
        valid_indices: 有效的索引列表（已排序）
    """
    map_file = directory / 'map.txt'
    with open(map_file, 'w', encoding='utf-8') as f:
        for idx in valid_indices:
            f.write(f'{idx}\n')
    print(f"  创建: map.txt (内容: {valid_indices})")


def process_scene(scene_dir: Path) -> None:
    """
    处理单个场景目录

    Args:
        scene_dir: 场景目录路径
    """
    print(f"\n处理场景: {scene_dir}")

    # 1. 提取floor_trav_i.png的索引
    valid_indices = extract_floor_indices(scene_dir, 'floor_trav_')

    if not valid_indices:
        print(f"  未找到任何floor_trav_i.png文件，跳过")
        return

    valid_indices_sorted = sorted(valid_indices)
    print(f"  找到的floor_trav索引: {valid_indices_sorted}")

    # 2. 过滤floors.txt
    filter_floors_txt(scene_dir, valid_indices)

    # 3. 删除无效的文件
    patterns_to_clean = ['meshcut_floor_', 'denoise_after_dilate_']
    remove_invalid_files(scene_dir, valid_indices, patterns_to_clean)

    # 4. 重命名所有文件
    all_patterns = ['floor_trav_', 'meshcut_floor_', 'denoise_after_dilate_']
    rename_files(scene_dir, valid_indices_sorted, all_patterns)

    # 5. 创建map.txt
    create_map_file(scene_dir, valid_indices_sorted)


def main():
    """
    主函数
    """
    if len(sys.argv) < 2:
        print("用法: python filter_floors.py <目录路径>")
        print("  目录路径可以是单个场景或包含多个场景的数据集")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"错误: {input_dir} 不是有效的目录")
        sys.exit(1)

    # 判断是场景还是数据集
    if is_scene_directory(input_dir):
        # 是单个场景，直接处理
        print(f"检测到单个场景: {input_dir}")
        process_scene(input_dir)
    else:
        # 是数据集，遍历所有子目录
        print(f"检测到数据集: {input_dir}")
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if not subdirs:
            print("未找到任何子目录")
            return

        print(f"找到 {len(subdirs)} 个子目录")
        for subdir in subdirs:
            process_scene(subdir)

    print("\n处理完成！")


if __name__ == '__main__':
    main()
