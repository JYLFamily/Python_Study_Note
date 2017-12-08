# coding:utf-8

import sys
import time
import logging
import tornado.web
import tornado.escape
from Project.LostRepair.FourthStep.ModelPlus import ModelPlus
from tornado.options import define, options
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(filename="my.log", filemode="a", level=logging.INFO)
define("port", default=7777, help="run on the given port", type=int)


class BaseHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(50)


class Server(BaseHandler):

    @run_on_executor()
    def post(self):
        time.sleep(3)
        model = ModelPlus(sys.argv[1])
        model.set_estimators()
        try:
            request_body = self.request.body
            return_proba = model.return_predict(request_body)
            self.write(return_proba)
        except Exception as e:
            logging.exception(e)

application = tornado.web.Application([
    (r"/evaluateScore", Server)
])


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    application.listen(options.port)
    logging.info("程序启动！！监听端口为：%s" % options.port)
    tornado.ioloop.IOLoop.instance().start()