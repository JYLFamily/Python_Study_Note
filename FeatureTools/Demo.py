# coding:utf-8

import pandas as pd
import featuretools as ft


class Demo(object):
    def __init__(self):
        self.__customers_df = ft.demo.load_mock_customer()["customers"]
        self.__sessions_df = ft.demo.load_mock_customer()["sessions"]
        self.__es = None
        self.__fm = None

    def set_es(self):
        self.__es = ft.EntitySet(id="customers")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="customers",
            index="customer_id",
            dataframe=self.__customers_df
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="sessions",
            index="session_id",
            dataframe=self.__sessions_df,
            variable_types={"device": ft.variable_types.Categorical}
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["customers"]["customer_id"],
                self.__es["sessions"]["customer_id"]
            )
        )

    def run_es(self):
        self.__fm, _ = ft.dfs(
            entityset=self.__es,
            target_entity="customers"
        )

        print(self.__fm.head())


if __name__ == "__main__":
    d = Demo()
    d.set_es()
    d.run_es()
