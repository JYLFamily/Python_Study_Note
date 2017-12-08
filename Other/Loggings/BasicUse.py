# coding:utf-8

import logging


class BasicUse(object):
    def __init__(self):
        logging.basicConfig(filename="example.log",
                            filemode="a",
                            format="[%(asctime)s]-[%(name)s]-[%(levelname)s]-[%(message)s]",
                            level=logging.DEBUG)

    def basic(self):
        logging.debug("This is Debug.")
        logging.info("This is Info.")


if __name__ == "__main__":
    bu = BasicUse()
    bu.basic()


