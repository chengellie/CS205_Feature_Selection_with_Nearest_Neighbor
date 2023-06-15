import csv

if __name__ == "__main__":
    file = "breast_cancer_data.csv"
    data = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            row = row[1:]
            data.append(row)
            if len(data) > 1:
                data[-1][0] = 0 if data[-1][0] == "B" else 1

    reformatted = "reformatted_" + file.replace("csv", "txt")
    with open(reformatted, "w") as csv_file:
        csv_file.writelines("  ".join(str(j) for j in i) + "\n" for i in data)
