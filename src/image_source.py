#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 14:39:11
@File	:   image_source.py
@Author :   Lavender1203
@desc   :   获取img目录下的所有图片
@version:   1.0
'''
import os

def get_img_sources():
    '''
    @desc: 用生成器的方式返回img目录下所有图片的路径
    @return: 图片路径
    @rtype: str
    '''
    # 支持的图片格式
    img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../img")
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if os.path.splitext(file)[1] in img_formats:
                yield os.path.join(root, file)