import logging
import os
from base.config import Config

"""
需求： 封装一个日志获取工具，用于项目中所有模块的日志处理。需要在控制台和文件中分类
思路：
1. 确保日志目录存在，不存在创建
2. 创建日志记录器: Logger
    2.1 获取Logger对象
    2.2 设置日志级别为所有控制器最低的(设置全局的日志级别)
3. 创建控制台控制器：StreamHandler
    3.1 创建控制台处理器对象
    3.2 设置日志级别为INFO
4. 创建文件处理器:FileHandler，并指定目录
    4.1 创建文件处理对象
    4.2 设置日志级别为DEBUG
5. 定义并设置日志格式：
    5.1 定义日志格式：logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    5.2 设置处理器日志格式
6. 把处理器添加到logger中

"""

def get_logger(project_name
               , level=logging.DEBUG
               , file_level=logging.DEBUG
               , console_level=logging.INFO
               , formatter='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(lineno)d - %(message)s'
               , encode='utf-8'
               , mode='a'
               , log_url='logs/app.log'):
    """
    团队通用日志打印工具
    :param project_name:     项目名，必填
    :param level:            全局日志级别...
    :param file_level:
    :param console_level:
    :param formatter: 格式
    :param encode: 编码
    :param mode: mode 类型
    :param log_url:  日志打印到哪个文件
    :return:
    """
    # 1. 确保日志目录存在，不存在创建

    # os.path.dirname(log_url) 获取文件所在的目录名
    # exist_ok=True: 如果文件夹存在, 不报错
    os.makedirs(os.path.dirname(log_url), exist_ok=True)

    # 2. 创建日志记录器: Logger
    # 2.1 获取Logger对象
    logger = logging.getLogger(project_name)
    # 2.2 设置日志级别为所有控制器最低的(设置全局的日志级别)
    logger.setLevel(level)  # 设置记录器级别

    # 3. 创建控制台控制器：StreamHandler
    # 3.1 创建控制台处理器对象
    console_handler = logging.StreamHandler()
    # 3.2 设置日志级别为INFO
    console_handler.setLevel(console_level)  # 控制台显示INFO及以上级别

    # 4. 创建文件处理器:FileHandler，并指定目录
    # 4.1 创建文件处理对象
    # TODO 需要注意，非特殊情况，mode一定不要写w
    file_handler = logging.FileHandler(log_url, mode=mode, encoding=encode)
    # 4.2 设置日志级别为DEBUG
    # TODO 实际工作中INFO就可以， DEBUG更多是用于本地调试代码的时候使用
    file_handler.setLevel(file_level)  # 文件记录DEBUG及以上级别

    # 5. 定义并设置日志格式：
    # 5.1 定义日志格式：eg: logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    formatter = logging.Formatter(fmt=formatter)

    # 5.2 设置处理器日志格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 6.避免重复添加
    if not logger.handlers:
        # 6.1 把处理器添加到logger中
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


single_logger = get_logger('project_RAG', log_url=Config().LOG_FILE)
