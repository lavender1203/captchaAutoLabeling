#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 16:31:06
@File	:   model.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''
import os
import sys
import ddddocr

from src.onnx import ONNX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.orientation import non_max_suppression, tag_images

# 建一个model抽象类
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from io import BytesIO

class BaseModel(ABC):
    @abstractmethod
    def load(self, model_path=None):
        pass

    @abstractmethod
    def run(self, image):
        pass

class DDDDModel(BaseModel):
    def __init__(self):
        self.model = None

    def load(self, model_path=None):
        if model_path:
            # 使用自己训练的模型
            self.model = ddddocr.DdddOcr(det=True, show_ad=False, import_onnx_path=model_path)
        else:
            # 默认使用ddddocr自带的模型
            self.model = ddddocr.DdddOcr(det=True, show_ad=False)

    def run(self, image):
        poses = self.model.detection(image)
        return poses
    
class OnnxModel(BaseModel):
    def __init__(self):
        self.model = None

    def load(self, model_path=None):
        self.model = ONNX(model_path)
        self.classes = ['icon', 'target']


    def run(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
        else:
            img = Image.open(file)
        img = img.convert('RGB')
        img = np.array(img)
        image_numpy = self.model.to_numpy(img, shape=(self.model.img_size, self.model.img_size))
        input_feed = self.model.get_input_feed(image_numpy)
        output = self.model.onnx_session.run(None, input_feed)[0]
        pred = non_max_suppression(output, 0.5, 0.5)
        res = tag_images(img, pred, self.model.img_size, 0.5)
        return res