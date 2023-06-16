import csv
import pandas as pd

if __name__ == "__main__":
    file = "breast_cancer_data.csv"
    df = pd.read_csv(file)
    df.drop("id", axis=1, inplace=True)
    df["diagnosis"].replace(["B", "M"], [0, 1], inplace=True)

    df[df.columns[1:]] = (
        df[df.columns[1:]].sub(df[df.columns[1:]].mean()).div(df[df.columns[1:]].std())
    )

    reformatted = "reformatted_" + file.replace("csv", "txt")
    df.to_csv(reformatted, sep=" ", index=False, header=False)
