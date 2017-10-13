import pandas as pd
import seaborn as sns


def show():
    raw_data = pd.read_csv("", \
                           header=None, sep="\t")

    sns.distplot(raw_data.iloc[:, 3])
    sns.plt.show()

if __name__ == "__main__":
    show()