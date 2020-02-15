import os
import logging
import sys
import math

# 初始化用来保存 log 类的set
logs = set()

def get_format():
    """
    如果不是在slurm集群上面运行的话, 就设定level为0
    总之就是生成一个formatter类, 用来构建输出的信息格式
    """
    format_str = '[%(asctime)s-%(filename)s#%(lineno)3d] [%(levelname)s] %(message)s'
    formatter = logging.Formatter(format_str)
    return formatter


def get_format_custom():
    format_str = '[%(asctime)s-%(message)s'
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level = logging.INFO, format_func=get_format):
    """
    初始化log类
    返回一个logger的类
    每次使用的时候, 都给予log元祖里面加一个(name, level)的子元祖, 用来防止重复初始化
    """
    # 防止重复初始化
    if (name, level) in logs: 
        return logging.getLogger(name)
    # 如果没有就将name和level放进去
    logs.add((name, level))
    logger = logging.getLogger(name) # 可以直接调用, 如果没有的话就自动创建一个.
    logger.setLevel(logging.DEBUG)           # 设定这个logger 的等级(在这个等级之上的才会被处理)
    ch = logging.StreamHandler()     # 初始化一个打印在命令行的handler
    ch.setLevel(level)               # 设定命令行的handler的级别
    formatter = format_func() # 构建format
    ch.setFormatter(formatter)       # 设定format
    logger.addHandler(ch)            # 为创建的logger增加handler
    return logger


def add_file_handler(name, log_file, level=logging.DEBUG):
    """
    为传入的类增加文件的handler
    """
    logger = logging.getLogger(name)    # 将目标的logger初始化一下, 如果没有就创建
    fh = logging.FileHandler(log_file, 'w+')  # 文件的名字, 初始化filehandler
    fh.setFormatter(get_format())  # setformat
    fh.setLevel(level)
    logger.addHandler(fh) # 添加handler


def print_speed(i, i_time, n, logger_name='global'):
    """
    用来生成目标的进度
    传入的时间按照秒来算
    print_speed(index, index_time, total_iteration, logger_name)
    """
    logger = logging.getLogger(logger_name)
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


# 在一开始先初始化global的logger
# init_log('global')


if __name__ == "__main__":
    """
    Usage
    """
    # 生成logger, 一开始自动初始化了一个global logger, 如果需要的话可以重新创建一个不重名的logger
    # 如果已经存在就直接返回创建好的logger
    logger_test= init_log('global', level=logging.INFO)
    # 将logger添加一个handler到文件
    add_file_handler("global", os.path.join('.', 'test.log'), level=logging.INFO)

    # log的方法
    logger_test.debug('this is a debug log')
    logger_test.info('hello info')
    logger_test.warning('this is a warning log')
    logger_test.error('this is a error message')
    logger_test.critical('this is critical')

    # 新增的方法 (默认用global来print)
    print_speed(1, 1, 10)