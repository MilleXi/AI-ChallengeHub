#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path

# 创建日志记录器
LOGGER = logging.getLogger("retail_shelf")

def colorstr(x, color='blue', bold=False, underline=False):
    """
    为字符串添加颜色/样式
    
    Args:
        x: 需要添加样式的字符串
        color: 颜色名称，可以是 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        bold: 是否粗体
        underline: 是否下划线
    
    Returns:
        带有ANSI颜色/样式的字符串
    """
    # 颜色代码
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
    }
    
    # 样式代码
    styles = ''
    if bold:
        styles += '\033[1m'
    if underline:
        styles += '\033[4m'
    
    # 重置代码
    reset = '\033[0m'
    
    # 返回带颜色的字符串
    return f"{styles}{colors.get(color, '')}{x}{reset}"

class EmojisHandler(logging.StreamHandler):
    """
    自定义日志处理器，用于在日志消息前添加表情符号
    """
    def __init__(self, stream=None):
        super().__init__(stream)
        self.emojis = {
            'INFO': '🔍',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨',
            'DEBUG': '🐞'
        }
    
    def emit(self, record):
        record.msg = f"{self.emojis.get(record.levelname, '')} {record.msg}"
        super().emit(record)

def setup_logger(name='retail_shelf', log_file=None, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果不为None，则同时输出到文件
        level: 日志级别
    
    Returns:
        logger: 日志记录器实例
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加表情符号控制台处理器
    console_handler = EmojisHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # 如果提供日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger 