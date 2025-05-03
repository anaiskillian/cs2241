#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
import math
import data_gen_utils

TEST_BASE_DIR = "/cs165/generated_data"
DOCKER_TEST_BASE_DIR = "/cs165/staff_test"


def generateDataMilestone2(dataSize):
    outputFile = f"{TEST_BASE_DIR}/data3_batch.csv"
    header_line = data_gen_utils.generateHeaderLine("db1", "tbl3_batch", 4)
    df = pd.DataFrame(
        np.random.randint(0, dataSize // 5, size=(dataSize, 4)),
        columns=["col1", "col2", "col3", "col4"],
    )
    df["col1"] = np.random.randint(0, 1000, size=(dataSize))
    df["col4"] = np.random.randint(0, 10000, size=(dataSize)) + df["col1"]
    df.to_csv(outputFile, sep=",", index=False, header=header_line, lineterminator="\n")
    return df


def generateDataMilestone3(dataSize):
    ctrl = f"{TEST_BASE_DIR}/data4_ctrl.csv"
    btree = f"{TEST_BASE_DIR}/data4_btree.csv"
    clustered = f"{TEST_BASE_DIR}/data4_clustered_btree.csv"
    header_ctrl = data_gen_utils.generateHeaderLine("db1", "tbl4_ctrl", 4)
    header_btree = data_gen_utils.generateHeaderLine("db1", "tbl4", 4)
    header_clustered = data_gen_utils.generateHeaderLine(
        "db1", "tbl4_clustered_btree", 4
    )

    df = pd.DataFrame(
        np.random.randint(0, dataSize // 5, size=(dataSize, 4)),
        columns=["col1", "col2", "col3", "col4"],
    )
    df["col1"] = np.random.randint(0, 1000, size=(dataSize))
    df["col4"] = np.random.randint(0, 10000, size=(dataSize)) + df["col1"]

    mask1 = np.random.rand(dataSize) < 0.05
    mask2 = np.random.rand(dataSize) < 0.02
    df["col2"] = np.random.randint(0, 10000, size=(dataSize))
    val1 = np.random.randint(0, dataSize // 5)
    val2 = np.random.randint(0, dataSize // 5)
    df.loc[mask1, "col2"] = val1
    df.loc[mask2, "col2"] = val2

    df.to_csv(ctrl, sep=",", index=False, header=header_ctrl, lineterminator="\n")
    df.to_csv(btree, sep=",", index=False, header=header_btree, lineterminator="\n")
    df.to_csv(
        clustered, sep=",", index=False, header=header_clustered, lineterminator="\n"
    )
    return val1, val2, df


class ZipfianDistribution:
    def __init__(self, zipfianParam, numElements):
        self.zipfianParam = zipfianParam
        self.numElements = numElements
        self.H_s = sum((1.0 / k**zipfianParam) for k in range(1, numElements + 1))

    def draw(self, unifSample):
        total = 0.0
        k = 0
        while unifSample >= total:
            k += 1
            total += (1.0 / k**self.zipfianParam) / self.H_s
        return k

    def sample(self, size):
        return np.vectorize(self.draw)(np.random.uniform(size=size))


def generateDataMilestone4(
    dataSizeFact,
    dataSizeDim1,
    dataSizeDim2,
    dataSizeSelect,
    zipfianParam,
    numDistinctElements,
):
    z = ZipfianDistribution(zipfianParam, numDistinctElements)
    output_files = {
        "fact": f"{TEST_BASE_DIR}/data5_fact.csv",
        "dim1": f"{TEST_BASE_DIR}/data5_dimension1.csv",
        "dim2": f"{TEST_BASE_DIR}/data5_dimension2.csv",
        "sel1": f"{TEST_BASE_DIR}/data5_selectivity1.csv",
        "sel2": f"{TEST_BASE_DIR}/data5_selectivity2.csv",
    }
    headers = {
        "fact": data_gen_utils.generateHeaderLine("db1", "tbl5_fact", 4),
        "dim1": data_gen_utils.generateHeaderLine("db1", "tbl5_dim1", 3),
        "dim2": data_gen_utils.generateHeaderLine("db1", "tbl5_dim2", 2),
        "sel1": data_gen_utils.generateHeaderLine("db1", "tbl5_sel1", 2),
        "sel2": data_gen_utils.generateHeaderLine("db1", "tbl5_sel2", 2),
    }

    df_fact = pd.DataFrame(
        np.random.randint(0, dataSizeFact // 5, size=(dataSizeFact, 4)),
        columns=["col1", "col2", "col3", "col4"],
    )
    df_fact["col1"] = z.sample(dataSizeFact)
    df_fact["col3"] = 1
    df_fact["col4"] = np.random.randint(1, dataSizeDim2, size=(dataSizeFact))
    df_fact.to_csv(
        output_files["fact"],
        sep=",",
        index=False,
        header=headers["fact"],
        lineterminator="\n",
    )

    df_dim1 = pd.DataFrame(
        np.random.randint(0, dataSizeDim1 // 5, size=(dataSizeDim1, 3)),
        columns=["col1", "col2", "col3"],
    )
    df_dim1["col1"] = z.sample(dataSizeDim1)
    df_dim1["col2"] = np.random.randint(1, dataSizeDim2, size=(dataSizeDim1))
    df_dim1.to_csv(
        output_files["dim1"],
        sep=",",
        index=False,
        header=headers["dim1"],
        lineterminator="\n",
    )

    df_dim2 = pd.DataFrame(
        np.random.randint(0, dataSizeDim2 // 5, size=(dataSizeDim2, 2)),
        columns=["col1", "col2"],
    )
    df_dim2["col1"] = np.arange(1, dataSizeDim2 + 1)
    df_dim2.to_csv(
        output_files["dim2"],
        sep=",",
        index=False,
        header=headers["dim2"],
        lineterminator="\n",
    )

    df_sel1 = pd.DataFrame(
        np.random.randint(0, dataSizeSelect // 5, size=(dataSizeSelect, 2)),
        columns=["col1", "col2"],
    )
    df_sel2 = pd.DataFrame(
        np.random.randint(0, dataSizeSelect // 5, size=(dataSizeSelect, 2)),
        columns=["col1", "col2"],
    )
    df_sel1.to_csv(
        output_files["sel1"],
        sep=",",
        index=False,
        header=headers["sel1"],
        lineterminator="\n",
    )
    df_sel2.to_csv(
        output_files["sel2"],
        sep=",",
        index=False,
        header=headers["sel2"],
        lineterminator="\n",
    )

    return df_fact, df_dim1, df_dim2, df_sel1, df_sel2


def generateDataMilestone5(dataSize):
    outputFile = f"{TEST_BASE_DIR}/data5.csv"
    header_line = data_gen_utils.generateHeaderLine("db1", "tbl5", 4)
    df = pd.DataFrame(
        np.random.randint(0, dataSize // 5, size=(dataSize, 4)),
        columns=["col1", "col2", "col3", "col4"],
    )
    df["col1"] = np.random.randint(0, 1000, size=(dataSize))
    df["col2"] = np.random.randint(0, 1000, size=(dataSize))
    df["col3"] = np.random.randint(0, 10000, size=(dataSize))
    df["col4"] = np.random.randint(0, 10000, size=(dataSize))
    df.to_csv(outputFile, sep=",", index=False, header=header_line, lineterminator="\n")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python data_gen.py <milestone> <dataSize> [additional args...]\n"
            "Milestones:\n"
            "  2 <dataSize>\n"
            "  3 <dataSize>\n"
            "  4 <dataSizeFact> <dataSizeDim1> <dataSizeDim2> <dataSizeSelect> <zipfianParam> <numDistinctElements>\n"
            "  5 <dataSize>"
        )
        sys.exit(1)

    milestone = int(sys.argv[1])

    if milestone == 2:
        generateDataMilestone2(int(sys.argv[2]))
    elif milestone == 3:
        generateDataMilestone3(int(sys.argv[2]))
    elif milestone == 4:
        if len(sys.argv) < 9:
            print("Milestone 4 requires 7 arguments.")
            sys.exit(1)
        generateDataMilestone4(
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            float(sys.argv[6]),
            int(sys.argv[7]),
        )
    elif milestone == 5:
        generateDataMilestone5(int(sys.argv[2]))
    else:
        print(f"Unknown milestone: {milestone}")
