#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 17:27:48
@File	:   onnx.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''
import time
import onnxruntime
import cv2
import numpy as np


class ONNX(object):
    def __init__(self, onnx_path, providers=None):
        '''初始化onnx'''
        if not providers:
            providers = ['CPUExecutionProvider']
        self.onnx_session=onnxruntime.InferenceSession(onnx_path, providers=providers)
        # print(onnxruntime.get_device())
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
        # self.classes=classes
        self.img_size = 640

    def get_input_name(self):
        '''获取输入节点名称'''
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        '''获取输出节点名称'''
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,image_tensor):
        '''获取输入tensor'''
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed
    
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        '''图片归一化'''
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def clip_coords(self,boxes, img_shape):
        '''查看是否越界'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
        :param img1_shape: 输入尺寸
        :param coords: 输入坐标
        :param img0_shape: 映射的尺寸
        :param ratio_pad:
        :return:
        '''

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
        coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
        coords[:, :4] /= gain  # 将box坐标对应到原始图像上
        self.clip_coords(coords, img0_shape)  # 边界检查
        return coords

    def to_numpy(self, img, shape):
        # 超参数设置
        img_size = shape
        # img_size = (640, 640)  # 图片缩放大小
        # 读取图片
        # src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        start = time.time()
        src_size = src_img.shape[:2]

        # 图片填充并归一化
        img = self.letterbox(src_img, img_size, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img = np.expand_dims(img, axis=0)
        # print('img resuming: ', time.time() - start)
        # 前向推理
        # start=time.time()

        return img