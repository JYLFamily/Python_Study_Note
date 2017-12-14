# coding:utf-8

import re
import os
import pandas as pd
from neo4j.v1 import GraphDatabase, basic_auth


class TwoDegreeRelationAddress(object):

    def __init__(self, output_path):
        self.__driver = GraphDatabase.driver("bolt://192.168.136.107:7687", auth=basic_auth("neo4j", "7Uclap0HR5QH"))
        self.__session = self.__driver.session()
        self.__phone_number = pd.read_csv("D:\\Project\\LostRepair\\Address\\qianzhandata", names=["phone_number"])
        self.__address_phone_number = dict()
        self.__output_path = output_path

    def search_neo4j_get_address(self):
        for lost_phone_number in self.__phone_number["phone_number"]:
            print("---search_neo4j_get_address---"+str(lost_phone_number))
            cypher = ("match "
                      "(t1:phone{phoneNo:\"" + str(lost_phone_number) + "\"})"
                      "-[r:has_self_tel|has_oper_tel]-"
                      "(u:user)"
                      "-[r2:has_d_contact_tel]-"
                      "(t2:phone)"
                      "return distinct t2.phoneNo as phoneNumber,r2.contactName as contactName")
            result = self.__session.run(cypher)

            self.__address_phone_number[lost_phone_number] = dict()
            for record in result:
                # record["contactName"] 可能不是一个 str 对象 , 如果不是 str 对象会报错 , 所以这里处理一下
                if isinstance(record["contactName"], str):
                    if re.search(r"爷|奶|姥|爸|妈|舅|叔|嫂|兄|弟|姐|妹", record["contactName"]):
                        self.__address_phone_number[lost_phone_number][record["phoneNumber"]] = []
                    else:
                        continue

    def search_neo4j_get_address_address(self):
        for lost_phone_number in self.__address_phone_number.keys():
            print("---search_neo4j_get_address_address---" + str(lost_phone_number))
            for one_repair_phone_number in self.__address_phone_number[lost_phone_number].keys():
                cypher = ("match "
                          "(t1:phone{phoneNo:\"" + str(one_repair_phone_number) + "\"})"
                          "-[r:has_self_tel|has_oper_tel]-"
                          "(u:user)"
                          "-[r2:has_d_contact_tel]-"
                          "(t2:phone)"
                          "return distinct t2.phoneNo as phoneNumber,r2.contactName as contactName")
                result = self.__session.run(cypher)
                for record in result:
                    if isinstance(record["contactName"], str):
                        if re.search(r"爷|奶|姥|爸|妈|舅|叔|嫂|兄|弟|姐|妹", record["contactName"]):
                            (self.__address_phone_number[lost_phone_number][one_repair_phone_number]
                             .append(record["phoneNumber"]))
                        else:
                            continue

    def count(self):
        df = []
        row = []
        for lost_phone_number in self.__address_phone_number.keys():
            # 失联电话
            row.append(lost_phone_number)
            # 失联电话一度修复出电话数目满足条件
            row.append(len(self.__address_phone_number[lost_phone_number].keys()))
            # 失联电话二度修复出电话数目满足条件
            two_repair_number = 0
            for one_repair_phone_number in self.__address_phone_number[lost_phone_number].keys():
                for two_repair_phone_number in self.__address_phone_number[lost_phone_number][one_repair_phone_number]:
                    diff = ([i for i in two_repair_phone_number
                             if i not in list(self.__address_phone_number[lost_phone_number].keys())])
                    two_repair_number += len(diff)
            row.append(two_repair_number)
            df.append(row)
            row = []
        (pd.DataFrame(df, columns=["Lost Phone", "One Relation", "Two Relation"])
             .to_csv(os.path.join(self.__output_path, "output2.csv"), index=False))


if __name__ == "__main__":
    tdra = TwoDegreeRelationAddress(output_path="C:\\Users\\Dell\\Desktop")
    tdra.search_neo4j_get_address()
    tdra.search_neo4j_get_address_address()
    tdra.count()
