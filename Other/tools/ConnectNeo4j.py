# coding:utf-8

import sys
import pandas as pd
from neo4j.v1 import GraphDatabase, basic_auth


class ConnectNeo4j(object):

    def __init__(self, input_path, output_path, offset):
        self.driver = GraphDatabase.driver("bolt://192.168.136.102:7687", auth=basic_auth("neo4j", "neo4jdev"))
        self.session = self.driver.session()
        self.input_path = input_path
        self.output_path = output_path
        self.input_df = pd.DataFrame
        self.id_list = []
        self.offset = offset
        self.offset_list = []
        self.decision_list = [False] * self.offset
        self.decision_df = pd.DataFrame
        self.output_df = pd.DataFrame

    def input_data(self):
        self.input_df = pd.read_csv(self.input_path, sep="\t", header=None)

    def set_id_list(self):
        for apply_id_no in self.input_df.loc[:, 0]:
            self.id_list.append(apply_id_no.split(sep=" ")[1])

    def search_neo4j(self, temp):
        print(temp)
        cypher = ("MATCH (u1:user {idNo:'" + str(temp[0]) + "'})-[r1]-(a:mobile)-[r2]-(u2:user {idNo:'" + str(temp[1]) + "'})\n"
                  "WITH u1, r1, a, u2, r2\n"
                  "MATCH (u1)-[r1]-(a)-[r3]-(u3:user{idNo:'" + str(temp[2]) + "'})\n"
                  "RETURN a")

        result = self.session.run(cypher)
        # 是否能够连接到一个节点上 , .run() 方法一定都能够返回一个 result , 但没有查到节点不会进行循环
        for record in result:
            print(record)
            return True
        else:
            return False

    def loop_search(self):
        for i in range(self.offset, len(self.id_list)):
            self.offset_list = []
            for j in range(self.offset+1):
                self.offset_list.append(self.id_list[i-j])
            self.decision_list.append(self.search_neo4j(self.offset_list))

    def output_data(self):
        self.decision_df = pd.Series(self.decision_list).to_frame()
        self.output_df = pd.concat([self.input_df, self.decision_df], axis=1)
        self.output_df.to_csv(self.output_path, sep="\t", index=False, header=False)

    def disconnect_neo4j(self):
        pass


if __name__ == "__main__":
    cn = ConnectNeo4j(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    cn.input_data()
    cn.set_id_list()
    cn.loop_search()
    cn.output_data()
    cn.disconnect_neo4j()