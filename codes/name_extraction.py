import os
import json

def rename_and_export(directory_path):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    name_dict = {}  # 创建一个空字典来存储原始文件名和编号

    for i, file in enumerate(sorted(files), start=1):
        old_path = os.path.join(directory_path, file)
        extension = os.path.splitext(file)[1]
        new_name = f"{i}{extension}"
        new_path = os.path.join(directory_path, new_name)

        os.rename(old_path, new_path)

        original_name = os.path.splitext(file)[0]  # 获取原始文件名，不包括扩展名
        name_dict[original_name] = str(i)  # 将原始文件名和编号添加到字典中

    # 导出为JSON
    with open('name_extracted.json', 'w') as json_file:
        json.dump(name_dict, json_file, indent=4)


# 指定要处理的文件夹路径
directory_path = 'original_dataset'

# 调用函数
rename_and_export(directory_path)
