#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 16:31:06
@File	:   model.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''
import importlib
import os
import sys
import ddddocr
from loguru import logger

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
        # TODO: 返回的结果需要统一
        pass

class DDDDModel(BaseModel):
    def __init__(self):
        self.model = None

    def load(self, model_path=None):
        if model_path is not None:
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
        # 是否配置了使用GPU推理，默认使用CPU
        use_gpu = os.environ.get("USE_GPU")
        is_use_gpu = use_gpu.lower() in ['true', '1', 'yes']
        if is_use_gpu:
            self.model = ONNX(model_path, ['CUDAExecutionProvider'])
        else:
            self.model = ONNX(model_path, ['CPUExecutionProvider'])

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
        poses = tag_images(img, pred, self.model.img_size, 0.5)
        poses = [pose['crop'] for pose in poses]
        return poses

class ModelFactory(object):
    models = {}

    @staticmethod
    def register(name, model):
        print(f"注册模型 {name}")
        ModelFactory.models[name] = model

    @staticmethod
    def get(name=None):
        if name is None:
            name = "DDDDModel"
        print(f"获取模型 {name}")
        return ModelFactory.models[name]
    
    @staticmethod
    def get_all_models():
        return ModelFactory.models

    @staticmethod
    def register_all_models(model_names):
        for name in model_names:
            model = ModelFactory.create_model(name)
            if model is not None:
                ModelFactory.register(name, model)

    @staticmethod
    def create_model(model_name):
        try:
            # 获取模型的类
            model_class = globals()[model_name]
            # 创建模型的实例
            model_instance = model_class()
            return model_instance
        except ImportError:
            logger.error(f"无法导入模型 {model_name}")
            return None

