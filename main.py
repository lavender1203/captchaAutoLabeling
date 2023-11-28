#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 13:36:51
@File	:   main.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''
from loguru import logger
from src.auto_detect import auto_detect

# 配置重定向路径
logger.add('auto_detect.log')

def main():
    # This function is intentionally left empty.
    auto_detect()

if __name__ == '__main__':
    main()
