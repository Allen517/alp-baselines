# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LogHandler.py
   Description :
   Author :       JHao
   date：          2017/3/6
-------------------------------------------------
   Change Activity:
                   2017/3/6: log handler
-------------------------------------------------
"""
__author__ = 'JHao'

import os

import logging

from logging.handlers import TimedRotatingFileHandler

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

# CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# ROOT_PATH = os.path.join(CURRENT_PATH, os.pardir)
# LOG_PATH = os.path.join(ROOT_PATH, 'log')

# CURRENT_PATH = os.getcwd()
ROOT_PATH = os.getcwd()
LOG_PATH = os.path.join(ROOT_PATH, 'log')
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

class LogHandler(logging.Logger):
    """
    LogHandler
    """

    def __init__(self, name, level=DEBUG):
        self.name = name
        self.level = level
        logging.Logger.__init__(self, self.name, level=level)
        self.__setFileHandler__()
        self.__setStreamHandler__()

    def __setFileHandler__(self, level=None):
        """
        set file handler
        :param level:
        :return:
        """
        file_name = os.path.join(LOG_PATH, '{name}.log'.format(name=self.name))
        # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
        file_handler = TimedRotatingFileHandler(filename=file_name, when='D', interval=1, backupCount=15)
        file_handler.suffix = '%Y%m%d.log'
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(self.level)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)


if __name__ == '__main__':
    log = LogHandler(__name__)
    log.info('this is a test msg')
    pass
