import random
import json
count = 200
dataList = []

for _ in range(count):
    a = round(random.random(), 2)
    b = round(random.random(), 2)

    # your rule: 1 only if values are exactly the same
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