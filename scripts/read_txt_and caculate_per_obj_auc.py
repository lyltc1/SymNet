import statistics
filename = "/home/Symnet/output/ycbv_pbr_16bit/YcbvkeyframeBest16bit_ycbv-test/aucadi_result.txt"  # Replace with the actual file path

data_dict = {}

with open(filename, "r") as file:
    lines = file.readlines()

for i in range(len(lines)):
    if lines[i].endswith("Object recalls:\n"):
        line = lines[i+1].strip()
        data = line.split(", ")
        for item in data:
            key, value = item.split(": ")
            data_dict.setdefault(int(key), []).append(float(value))
print(filename)

for k, v in data_dict.items():
    average = statistics.mean(v)
    print(average)

for k, v in data_dict.items():
    print(k, "------------")
    for i in v:
        print(i, end=" ")
    print("\n")

