#!/usr/bin/env python3
import os
import argparse
import shutil


def copy_jsons(d1: str, d2: str):
    d1 = os.path.abspath(d1)
    d2 = os.path.abspath(d2)

    if not os.path.isdir(d1):
        raise NotADirectoryError(f"{d1} 不是有效目录")
    if not os.path.isdir(d2):
        raise NotADirectoryError(f"{d2} 不是有效目录")

    json_files = [f for f in os.listdir(d1) if f.endswith(".json")]

    for jf in json_files:
        json_path = os.path.join(d1, jf)
        scene_name = os.path.splitext(jf)[0]  # 去掉 .json

        target_dir = os.path.join(d2, scene_name)

        if not os.path.isdir(target_dir):
            print(f"[WARN] 在 {d2} 下找不到目录: {scene_name}，跳过。")
            continue

        dst_path = os.path.join(target_dir, jf)
        shutil.copy2(json_path, dst_path)
        print(f"[OK] {json_path} → {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="把 d1 下所有 .json 文件复制到 d2 下同名子目录中"
    )
    parser.add_argument("d1", help="包含 .json 文件的目录")
    parser.add_argument("d2", help="包含场景子目录的目录")
    args = parser.parse_args()

    copy_jsons(args.d1, args.d2)


if __name__ == "__main__":
    main()
