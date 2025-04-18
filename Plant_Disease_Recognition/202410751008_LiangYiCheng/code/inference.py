import os
from ultralytics import YOLO

model = YOLO("./code/best-train.pt", task="detect")
dir = "./data_example/input"
files_abs_path_arr = []


def traverse_files(dir):
    global files_abs_path_arr
    files = os.listdir(dir)
    for file in files:
        # 判断是否是文件夹
        if os.path.isdir(file):
            traverse_files(dir=file)
        else:
            file_abs_path = os.path.join(dir, file)
            files_abs_path_arr.append(file_abs_path)


traverse_files(dir)
for i in range(0, len(files_abs_path_arr)):
    result = model.predict(files_abs_path_arr[i])
    result[0].save(f"./data_example/output/{i}.jpg")
