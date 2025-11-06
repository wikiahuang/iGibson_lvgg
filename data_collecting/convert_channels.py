import os
import argparse
import re
from PIL import Image
import sys

def convert_images_to_single_channel(root_dir):
    """
    遍历指定根目录下的所有子目录，找到所有符合 'floor_trav_[数字].png' 模式的图片，
    并将其从4通道（RGBA）转换为单通道（灰度图）。

    Args:
        root_dir (str): 要处理的根目录路径。
    """
    # 使用正则表达式定义文件名模式：
    # ^                  - 匹配字符串的开头
    # floor_trav_        - 匹配字面文本 "floor_trav_"
    # \d+                - 匹配一个或多个数字
    # \.png              - 匹配 ".png"
    # $                  - 匹配字符串的结尾
    pattern = re.compile(r'^floor_trav_\d+\.png$')
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    found_files = 0

    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在或不是一个有效的目录。")
        sys.exit(1)

    print(f"开始在 '{root_dir}' 及其子目录中搜索符合 'floor_trav_[数字].png' 模式的图片...")
    err = []
    # os.walk 会递归地遍历所有子目录
    # dirpath 是当前目录路径, _, filenames 是该目录下的文件列表
    for dirpath, _, filenames in os.walk(root_dir):
        # 遍历当前目录下的所有文件名
        #print("dirpath: ", dirpath)
        for filename in filenames:
            # 检查文件名是否符合我们定义的模式
            #print("filename: ", filename)
            if pattern.match(filename):
                found_files += 1
                image_path = os.path.join(dirpath, filename)
                #print(f"\n找到匹配的图片: {image_path}")
                
                try:
                    # 打开图片
                    with Image.open(image_path) as img:
                        # 检查图片是否为4通道 (RGBA)
                        if img.mode == 'RGBA':
                            #print(f"  - 原始模式: {img.mode}, 尺寸: {img.size}")
                            
                            # 使用 .convert('L') 方法转换为单通道灰度图
                            # 'L' 模式代表 8-bit pixels, black and white
                            gray_img = img.convert('L')
                            
                            # 直接覆盖保存原文件
                            gray_img.save(image_path)
                            
                            #print(f"  - 成功转换并保存为单通道图片。")
                            processed_count += 1
                        else:
                            print(f"  - 跳过，该图片不是4通道 (当前模式: {img.mode})。")
                            err.append(image_path)
                            skipped_count += 1

                except Exception as e:
                    print(f"  - 处理失败: {image_path}. 错误: {e}")
                    error_count += 1
    
    if found_files == 0:
        print("\n在指定目录中没有找到任何符合模式 'floor_trav_[数字].png' 的文件。")

    print("\n--------------------")
    print("处理完成！")
    print(f"总计成功处理: {processed_count} 个文件")
    print(f"总计跳过 (非4通道): {skipped_count} 个文件")
    print(f"总计出错: {error_count} 个文件")
    for png in err:
        print("wrong_path: ", png)
    print("--------------------")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="将指定目录下所有子目录中符合 'floor_trav_[数字].png' 模式的图片从4通道转换为单通道。"
    )
    parser.add_argument(
        "directory", 
        type=str, 
        help="要处理的根目录路径。"
    )
    
    args = parser.parse_args()
    
    # 调用主函数，传入用户提供的目录
    convert_images_to_single_channel(args.directory)