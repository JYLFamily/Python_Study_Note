# coding:utf-8
import sys
import pandas as pd
from neo4j.v1 import GraphDatabase, basic_auth


driver = GraphDatabase.driver("bolt://192.168.136.102:7687", auth=basic_auth("neo4j", "neo4jdev"))
session = driver.session()


def load():
    df = pd.read_csv(sys.argv[1], sep="\t", header=None)
    idno_list = []
    for apply_id_no in df.loc[:, 0]:
        idno_list.append(apply_id_no.split(sep=" ")[1])

    return idno_list


def search_neo4j(variable):
    temp = 0
    print(variable[0])
    for i in variable[1:]:
        cypher = "MATCH" \
                 "(n1:user {idNo:'" + str(variable[0]) + "'})" \
                   "-[r1]-" \
                 "(n)" \
                   "-[r2]-" \
                 "(n2:user {idNo:'" + str(i) + "'}) " \
                 "RETURN COUNT(*) as number"
        result = session.run(cypher)
        for record in result:
            temp += record["number"]

    return True if temp > 0 else False


def loop_search(idno_list, offset):
    decision = [False] * offset
    for i in range(offset, len(idno_list)):
        temp = []
        for j in range(offset+1):
            temp.append(idno_list[i - j])
        decision.append(search_neo4j(temp))

    return decision


if __name__ == "__main__":
    idno_list = load()
    decision = loop_search(idno_list, offset=int(sys.argv[3]))
    decision = pd.Series(decision).to_frame()
    df = pd.read_csv(sys.argv[1], sep="\t", header=None)
    df = pd.concat([df, decision], axis=1)
    df.columns = list(range(0, df.shape[1]))
    df = df.loc[(df[df.shape[1] - 1] == True), list(range(df.shape[1] - 1))]
    df.to_csv(sys.argv[2], sep="\t", header=False, index=False)