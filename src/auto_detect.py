#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 13:38:31
@File	:   auto_detect.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''

import time
import cv2
import os
from loguru import logger
from src import image_source
from tqdm import tqdm
from src.model import DDDDModel, ModelFactory
from src.model import OnnxModel

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

def get_poses_with_model(pic, model):
    res = []
    train_res = []
    train_str = ""

    with open(pic, 'rb') as f:
        image = f.read()
    poses = model.run(image)

    im = cv2.imread(pic)
    w = im.shape[1]
    h = im.shape[0]

    target_cnt = 0
    icon_cnt = 0
    for box in poses:
        box = box.get('crop') if 'crop' in box else box
        x1, y1, x2, y2 = box
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # 过滤出模板 模板数量等于识别出的数量才可能成功
        x = int((x1 + x2)/2)
        y = int((y1 + y2)/2)
        if y2 < 360:
            res.append([x, y])
            icon_cnt = icon_cnt + 1
            train_res.append([0, round_6(x/w), round_6(y/h), round_6((x2-x1)/w), round_6((y2-y1)/h)])
            train_str = train_str + f"0 {round_6(x/w)} {round_6(y/h)} {round_6((x2-x1)/w)} {round_6((y2-y1)/h)}\n"
        else:
            target_cnt = target_cnt + 1
            train_res.append([1, round_6(x/w), round_6(y/h), round_6((x2-x1)/w), round_6((y2-y1)/h)])
            train_str = train_str + f"1 {round_6(x/w)} {round_6(y/h)} {round_6((x2-x1)/w)} {round_6((y2-y1)/h)}\n"
    if icon_cnt == target_cnt:
        # 目标检测成功
        # logger.debug(train_res)
        # 获取pic的文件名, 用于保存
        filename = os.path.basename(pic)
        cv2.imwrite(f"{project_path}/result/success/{filename}", im)
        return True, train_str, res
    else:
        # 获取pic的文件名, 用于保存
        filename = os.path.basename(pic)
        cv2.imwrite(f"{project_path}/result/fail/{filename}", im)
        return False, train_str, res

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
    paths = [project_path + "/data/images/", project_path + "/data/labels/", project_path + "/result/success/", project_path + "/result/fail/"]
    for path in paths:
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
    for path in paths:
        os.mkdir(path)
    # 添加处理进度条
    image_sources_list = list(image_sources)
    tdqm_image_sources = tqdm(image_sources_list)    
    for filename in tdqm_image_sources:
        total_count = total_count + 1
        flag, train_str, _ = get_poses_with_model(filename, model=model)
        # 目标检测成功率保留两位小数
        success_rate = round(count/total_count*100,2)
        tdqm_image_sources.set_description(f"目标检测成功率: {success_rate}%")
        if not flag:
            # logger.debug(f"{filename} 目标检测失败")
            pass
        else:
            count = count + 1
            # 保存文件到另一个目录, 用于训练
            name = os.path.splitext(filename)[0].split("/")[-1]
            # 文件后缀
            suffix = os.path.splitext(filename)[1]
            new_filename = project_path + "/data/images/" + name + suffix
            with open(new_filename, "wb") as f:
                f.write(open(filename, "rb").read())
            # 保存训练数据
            new_filename = project_path + "/data/labels/" + name + ".txt"
            with open(new_filename, "w") as f:
                f.write(train_str)
    # 识别成功率
    logger.debug(f"识别成功率: {round_6(count/total_count)*100}%")


