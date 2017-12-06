# coding:utf-8

import logging


class FirstStep(object):

    def __init__(self):
        # 设置 logger 名称
        self.__logger = logging.getLogger(__name__)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.__file_handler = None
        self.__console_handler = None

    def use_logging_with_control(self):
        # 设置写文件 handler 默认 mode="a"
        self.__file_handler = logging.FileHandler("my.log", mode="a")
        self.__file_handler.setLevel(logging.INFO)
        self.__file_handler.setFormatter(self.__formatter)

        # 设置写控制台 handler
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.DEBUG)
        self.__console_handler.setFormatter(self.__formatter)

        self.__logger.addHandler(self.__file_handler)
        self.__logger.addHandler(self.__console_handler)

    def use_logging_output_console(self):
        self.__logger.debug("This is debug message")
        self.__logger.info("This is info message")
        self.__logger.warning("This is warning message")
        self.__logger.error("This is error message")
        self.__logger.critical("This is critical message")


if __name__ == "__main__":
    fs = FirstStep()
    fs.use_logging_with_control()
    fs.use_logging_output_console()
