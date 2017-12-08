# coding:utf-8

import logging


class TryExceptLogging(object):

    def __init__(self, number):
        logging.basicConfig(filename="TryExceptLogging.log",
                            filemode="a",
                            format="[%(asctime)s]-[%(name)s]-[%(levelname)s]-[%(message)s]",
                            level=logging.DEBUG)
        self.__number = number

    def test_try_except(self):
        try:
            print(10 / self.__number)
        except Exception as e:
            logging.exception(e)


if __name__ == "__main__":
    tel = TryExceptLogging(0)
    tel.test_try_except()