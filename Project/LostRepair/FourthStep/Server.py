# coding:utf-8

import logging
import tornado.web
import tornado.escape
from tornado.options import define, options
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from Project.LostRepair.FourthStep.FourthStep import FourthStep
logger = logging.getLogger("justdoit_api.py")
define("port", default=7777, help="run on the given port", type=int)


class BaseHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(50)


class Server(BaseHandler):

    @run_on_executor()
    def post(self):
        estimator = FourthStep("C:\\Users\\Dell\\Desktop\\week")
        estimator.set_estimators()
        try:
            request_body = self.request.body
            print(request_body)
            return_proba = estimator.return_predict(request_body)
            print(return_proba)
            # 模型处理完毕返回json
            # respon_json = tornado.escape.json_encode(return_proba)
            self.write(return_proba)
        except Exception  as e:
            pass
        finally:
            # self.finish()
            pass

application = tornado.web.Application([
    (r"/evaluateScore", Server)
])


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    application.listen(options.port)
    logger.info('程序启动！！监听端口为：%s' % options.port)
    tornado.ioloop.IOLoop.instance().start()