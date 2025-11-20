#exapmle data generation script for generating training data
# generates 200 data points where the input is two random numbers between 0 and 1
# the output is 1 if the numbers are equal, else 0

import random
import json
count = 200
dataList = []

for _ in range(count):
    a = round(random.random(), 2)
    b = round(random.random(), 2)

    output = 1 if a == b else 0

    data = {
        "input": [a, b],
        "output": [output]
    }

    dataList.append(data)

for d in dataList:
    print(d)


with open("trainingData.json", "w") as f:
    json.dump([], f, indent=4)

with open("trainingData.json", "w") as f:
    json.dump(dataList, f, indent=4)