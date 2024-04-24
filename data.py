import os
import shutil
from tqdm import tqdm
import pandas as pd

# 读取CSV表格,把对应的图片放到对应的文件夹中

df = pd.read_csv('data/DB.csv', sep=';')

label_list = {}

# # 遍历标签类别
for row in df.itertuples():
    label = getattr(row, 'Plane')
    if label in label_list.keys():
        label_list[label] += 1
    else:
        label_list[label] = 1

print(label_list)

# for k in label_list.keys():
#     img_f_path = f'data/data/{k}'
#     if not os.path.exists(img_f_path):
#         # 新建文件夹
#         os.mkdir(img_f_path)

# 复制图片
# for row in tqdm(df.itertuples()):
#     label = getattr(row, 'Plane')
#     f = getattr(row, 'Image_name') + '.png'
#
#     path_from = 'data/Images/' + f
#     path_to = 'data/data/' + label+'/'+f
#     shutil.copyfile(path_from, path_to)
