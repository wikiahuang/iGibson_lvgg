#!/usr/bin/env python3
"""
脚本功能：
将input_path下的json文件复制到output_path下的同名子目录中
例如：n1.json -> output_path/n1/n1.json
"""

import sys
import shutil
from pathlib import Path


def copy_json_files(input_path: Path, output_path: Path) -> None:
    """
    将输入目录中的json文件复制到输出目录的同名子目录中

    Args:
        input_path: 输入目录路径，包含多个json文件
        output_path: 输出目录路径，包含多个子目录
    """
    # 获取所有json文件
    json_files = list(input_path.glob('*.json'))

    if not json_files:
        print(f"在 {input_path} 中未找到任何json文件")
        return

    print(f"找到 {len(json_files)} 个json文件")

    # 处理每个json文件
    success_count = 0
    skip_count = 0

    for json_file in json_files:
        # 获取文件名（不含扩展名）
        name_without_ext = json_file.stem

        # 查找目标子目录
        target_subdir = output_path / name_without_ext

        if not target_subdir.exists() or not target_subdir.is_dir():
            print(f"跳过 {json_file.name}: 目标子目录 {target_subdir} 不存在")
            skip_count += 1
            continue

        # 目标文件路径
        target_file = target_subdir / json_file.name

        # 复制文件（如果存在则覆盖）
        try:
            shutil.copy2(json_file, target_file)
            if target_file.exists():
                print(f"✓ 复制: {json_file.name} -> {target_subdir.name}/{json_file.name}")
                success_count += 1
        except Exception as e:
            print(f"✗ 错误: 复制 {json_file.name} 失败: {e}")

    # 打印统计信息
    print(f"\n处理完成！")
    print(f"  成功复制: {success_count} 个文件")
    print(f"  跳过: {skip_count} 个文件")


def main():
    """
    主函数
    """
    if len(sys.argv) < 3:
        print("用法: python copy_json_to_subdirs.py <input_path> <output_path>")
        print("  input_path: 包含json文件的输入目录")
        print("  output_path: 包含子目录的输出目录")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # 验证输入目录
    if not input_path.exists() or not input_path.is_dir():
        print(f"错误: 输入路径 {input_path} 不是有效的目录")
        sys.exit(1)

    # 验证输出目录
    if not output_path.exists() or not output_path.is_dir():
        print(f"错误: 输出路径 {output_path} 不是有效的目录")
        sys.exit(1)

    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print()

    # 执行复制操作
    copy_json_files(input_path, output_path)


if __name__ == '__main__':
    main()
