#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on  2023/11/28 13:36:51
@File	:   main.py
@Author :   Lavender1203
@desc   :   None
@version:   1.0
'''
import os
from dotenv import load_dotenv
from loguru import logger
from src.auto_detect import auto_detect

logger.remove()
# 配置重定向路径
logger.add('auto_detect.log')

def set_log_level():
    # 配置日志级别
    is_debug = os.environ.get("DEBUG").lower() in ['true', '1', 'yes']
    if is_debug:
        logger.level('DEBUG')
    else:
        logger.level('ERROR')

def main():
    load_dotenv(verbose=True)
    set_log_level()
    # 注册model
    
    auto_detect()

if __name__ == '__main__':
    main()
