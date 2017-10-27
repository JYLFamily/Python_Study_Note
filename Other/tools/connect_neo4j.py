# coding:utf-8
import numpy as np
import pandas as pd
from neo4j.v1 import GraphDatabase, basic_auth


driver = GraphDatabase.driver("bolt://192.168.136.102:7687", auth=basic_auth("neo4j", "neo4jdev"))
session = driver.session()


def load():
    df = pd.read_csv("C:\\Users\\Dell\\Desktop\\output", sep="\t", header=None)
    idno_list = []
    for apply_id_no in df.loc[:, 0]:
        idno_list.append(apply_id_no.split(sep=" ")[1])

    return idno_list


def search_neo4j(variable):
    temp = 0
    print(variable[0])
    for i in variable[1:]:
        cypher = "match " \
                 "(n1:user {idNo:'" + str(variable[0]) + "'})" \
                   "-[r1:has_self_tel]-" \
                 "(m1:phone)" \
                   "-[r2]-" \
                 "(n2:user {idNo:'" + str(i) + "'}) " \
                 "return count(*) as number"
        result = session.run(cypher)
        for record in result:
            temp += record["number"]

    return True if temp > 0 else False


def loop_search(idno_list, offset):
    decision = [False] * offset
    for i in np.arange(start=offset, stop=len(idno_list)):
        temp = []
        for j in range(offset+1):
            temp.append(idno_list[i - j])
        decision.append(search_neo4j(temp))

    return decision


if __name__ == "__main__":
    idno_list = load()
    decision = loop_search(idno_list, offset=3)
    decision = pd.Series(decision).to_frame()
    df = pd.read_csv("C:\\Users\\Dell\\Desktop\\output", sep="\t", header=None)
    df = pd.concat([df, decision], axis=1)
    df.to_csv("C:\\Users\\Dell\\Desktop\\output_label", sep="\t", header=False, index=False)