# coding:utf-8

import sys
import pandas as pd
from neo4j.v1 import GraphDatabase, basic_auth


class ConnectNeo4jCount(object):

    def __init__(self, input_path, output_path, offset):
        self.driver = GraphDatabase.driver("bolt://192.168.136.102:7687", auth=basic_auth("neo4j", "neo4jdev"))
        self.session = self.driver.session()
        self.input_path = input_path
        self.output_path = output_path
        self.input_df = pd.DataFrame
        self.id_list = []
        self.offset = offset
        self.offset_list = []
        self.decision_list = [0] * self.offset
        self.decision_df = pd.DataFrame
        self.output_df = pd.DataFrame

    def input_data(self):
        self.input_df = pd.read_csv(self.input_path, sep="\t", header=None)

    def set_id_list(self):
        for apply_id_no in self.input_df.loc[:, 0]:
            self.id_list.append(apply_id_no.split(sep=" ")[1])

    def search_neo4j(self, temp):
        print(temp[0], end=" ")
        counter = 0
        for i in temp[1:]:
            cypher = ("MATCH"
                      "(n1:user {idNo:'" + str(temp[0]) + "'})"
                      "-[r1]-"
                      "(n:mobile)"
                      "-[r2]-"
                      "(n2:user {idNo:'" + str(i) + "'}) "
                      "RETURN COUNT(*) as number")
            result = self.session.run(cypher)
            for record in result:
                counter += 1 if record["number"] > 0 else 0
        print(counter)
        return counter

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
    cn = ConnectNeo4jCount(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    cn.input_data()
    cn.set_id_list()
    cn.loop_search()
    cn.output_data()
    cn.disconnect_neo4j()