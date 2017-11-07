# coding:utf-8

import sys
import json
import pandas as pd


class JsonResolve(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.list_list = []
        self.data_frame = None

    def json_to_list(self):
        with open(self.input_path, "r", encoding="UTF-8") as f:
            line = f.readline()
            while line:
                self.list_list.append(list(json.loads(line)["data"][0].values()))
                line = f.readline()
                line = line.strip()
                print(line)

    def list_to_frame(self):
        self.data_frame = pd.DataFrame(self.list_list)

    def frame_to_csv(self):
        self.data_frame.to_csv(self.output_path, sep="\t", index=False)


if __name__ == "__main__":
    jr = JsonResolve(sys.argv[1], sys.argv[2])
    jr.json_to_list()
    jr.list_to_frame()
    jr.frame_to_csv()

