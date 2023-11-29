#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 13:38:31
@File	:   auto_detect.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''

from io import BytesIO
import time
import cv2
import os
from loguru import logger
import numpy as np
from src import image_source
from tqdm import tqdm
from src.model import  ModelFactory
import yaml


# 获取文件绝对路径
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

def round_6(num):
    return round(num, 6)

def get_coordinates(poses):
    # 获取中心点坐标
    res = []
    for box in poses:
        x1, y1, x2, y2 = box
        x = int((x1 + x2)/2)
        y = int((y1 + y2)/2)
        res.append([x, y])
    return res

def get_min_pose_size(poses):
    min_width = min_height = float('inf')

    for pose in poses:
        x1, y1, x2, y2 = pose
        width = x2 - x1
        height = y2 - y1
        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height

    return min_width, min_height

def find_in_top(poses, threshold, is_top=True):
    icons = []
    targets = []
    # 求出最大的y2和最小的y1
    max_y = max([pose[3] for pose in poses])
    min_y = min([pose[1] for pose in poses])
    if is_top:
        for pose in poses:
            y_change = max_y - pose[3]
            if y_change > threshold:
                icons.append(pose)
            else:
                targets.append(pose)
    else:
        for pose in poses:
            y_change = pose[1] - min_y
            if y_change > threshold:
                icons.append(pose)
            else:
                targets.append(pose)
    return icons, targets

def split_poses_by_y_change(poses):
    """
    按y值变化大小分类，模板应该都靠近顶部或者底部
    """
    icons = []
    targets = []
    _, h = get_min_pose_size(poses)
    threshold = h*0.5 # 最小图片的一半高 
    # 在顶部查找
    icons, targets = find_in_top(poses, threshold=threshold)
    # 判断列表是否为空
    if targets:
        return icons, targets
    # 在底部查找
    icons, targets = find_in_top(poses, threshold=threshold, is_top=False)
    
    return icons, targets

def gen_yolov5_yaml(path, nc):
    data = dict(
        path=path,
        train="images/train",
        val="images/val",
        test="images/test",
        nc=nc,
        names=str([i for i in range(nc)])
    )

    with open(f'{project_path}/data/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def get_poses_with_model(pic, model):
    train_res = []
    train_str = ""

    with open(pic, 'rb') as f:
        image = f.read()
    poses = model.run(image)

    # image转换为cv2 imread后的格式
    im = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    w = im.shape[1]
    h = im.shape[0]

    # 用于判断是否存在模板，如果y坐标变化不大，集中在图片顶部或底部，那么就是模板targets
    icons, targets = split_poses_by_y_change(poses)
    icon_class = 0
    target_class = 1
    icon_cnt = len(icons)
    target_cnt = len(targets)
    if target_cnt == 0:
        # 生成yolov5.yaml
        nc = 1
    else:
        nc = 2
    dataset_path = os.environ.get("DATASET_PATH")
    gen_yolov5_yaml(dataset_path, nc)
    
    for icon in icons:
        x1, y1, x2, y2 = icon
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # 左上角画个很小的数字
        im = cv2.putText(im, str(icon_class), (x1, y1+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
        # 过滤出模板 模板数量等于识别出的数量才可能成功
        x = int((x1 + x2)/2)
        y = int((y1 + y2)/2)

        train_res.append([0, round_6(x/w), round_6(y/h), round_6((x2-x1)/w), round_6((y2-y1)/h)])
        train_str = train_str + f"{icon_class} {round_6(x/w)} {round_6(y/h)} {round_6((x2-x1)/w)} {round_6((y2-y1)/h)}\n"
            
    for target in targets:
        x1, y1, x2, y2 = target
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # 左上角画个很小的数字
        im = cv2.putText(im, str(target_class), (x1, y1+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
        # 过滤出模板 模板数量等于识别出的数量才可能成功
        x = int((x1 + x2)/2)
        y = int((y1 + y2)/2)
        train_res.append([1, round_6(x/w), round_6(y/h), round_6((x2-x1)/w), round_6((y2-y1)/h)])
        train_str = train_str + f"{target_class} {round_6(x/w)} {round_6(y/h)} {round_6((x2-x1)/w)} {round_6((y2-y1)/h)}\n"

    if target_cnt == 0 or (icon_cnt == target_cnt and target_cnt != 0):
        # 目标检测成功
        # logger.debug(train_res)
        # 获取pic的文件名, 用于保存
        filename = os.path.basename(pic)
        cv2.imwrite(f"{project_path}/result/success/{filename}", im)
        return True, train_str
    else:
        # 获取pic的文件名, 用于保存
        filename = os.path.basename(pic)
        cv2.imwrite(f"{project_path}/result/fail/{filename}", im)
        return False, train_str

def load_model():
    model_type = os.environ.get("MODEL_TYPE")
    # 使用依赖注入方式重构代码, 使得可以自由切换模型
    model_path = os.environ.get("MODEL_PATH")
    if model_path and model_type:
        model_path = project_path + model_path
        model = ModelFactory.get(name=model_type)
        model.load(model_path=model_path)
    else:
        model = ModelFactory.get()
        model.load()
    if not model:
        raise ValueError("模型加载失败")
    return model

def export_yolov5_dataset(train_dataset):
    split_rate = os.environ.get("SPLIT_RATE")
    split_cnt = int(len(train_dataset) * float(split_rate))
    for i, train_data in enumerate(train_dataset):
        if i <= split_cnt:
            train_name = "train"
        else:
            train_name = "val"
        filename = train_data["filename"]
        train_str = train_data["train_str"]
        # 保存文件到另一个目录, 用于训练
        name = os.path.splitext(filename)[0].split("/")[-1]
        # 文件后缀
        suffix = os.path.splitext(filename)[1]
        new_filename = project_path + f"/data/images/{train_name}/" + name + suffix
        with open(new_filename, "wb") as f:
            f.write(open(filename, "rb").read())
        # 保存训练数据
        new_filename = project_path + f"/data/labels/{train_name}/" + name + ".txt"
        with open(new_filename, "w") as f:
            f.write(train_str)

def auto_detect():
    # 加载模型
    model = load_model()
    # 遍历img目录下的所有图片
    count = 0
    total_count = 0
    image_sources = image_source.get_img_sources()
    # 判断是否有图片
    if not any(image_sources):
        logger.debug("img目录下没有图片, 请添加图片后重试")
        return
    # 清空文件夹
    paths = [project_path + "/data/images/train/",project_path + "/data/images/val/", project_path + "/data/labels/train/",project_path + "/data/labels/val/", project_path + "/result/success/", project_path + "/result/fail/"]
    for path in paths:
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
    for path in paths:
        os.mkdir(path)
    # 添加处理进度条
    image_sources_list = list(image_sources)
    tdqm_image_sources = tqdm(image_sources_list)  
    train_dataset = []  
    for filename in tdqm_image_sources:
        total_count = total_count + 1
        flag, train_str = get_poses_with_model(filename, model=model)
        # 目标检测成功率保留两位小数
        success_rate = round(count/total_count*100,2)
        tdqm_image_sources.set_description(f"目标检测成功率: {success_rate}%")
        if not flag:
            # logger.debug(f"{filename} 目标检测失败")
            pass
        else:
            count = count + 1
            train_data = dict(train_str=train_str, filename=filename)
            train_dataset.append(train_data)
    export_yolov5_dataset(train_dataset)
    # 识别成功率
    logger.debug(f"识别成功率: {round_6(count/total_count)*100}%")


