# 统计所有图片的尺寸大小


import os
from PIL import Image
from collections import Counter

# 定义文件夹路径
folder_path = 'data/Images'  # 请替换为你的文件夹路径

# 创建一个字典，用于统计分辨率和对应图片数量
resolution_counter = Counter()

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为图片文件（根据文件后缀名）
    if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff')):
        # 构建文件路径
        file_path = os.path.join(folder_path, filename)
        # 打开图片
        with Image.open(file_path) as img:
            # 获取图片的分辨率（尺寸）
            resolution = img.size  # 返回 (width, height)
            # 将分辨率添加到计数器中
            resolution_counter[resolution] += 1

# 按照图片数量排序
sorted_resolutions = resolution_counter.most_common()

# 计算总图片数量
total_images = sum(resolution_counter.values())

# 输出结果
print("分辨率\t图片数量\t占比")
for resolution, count in sorted_resolutions:
    percentage = (count / total_images) * 100
    print(f"{resolution}\t{count}\t{percentage:.2f}%")
