import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

dir_path = '/home/Symnet/output/tless_pbr_only_results'

data = []
count = 0

for root, dirnames, filenames in os.walk(dir_path):
    for filename in filenames:
        if filename == "scores_bop19.json":
            f = os.path.join(root, filename)
            with open(f, 'r') as file:
                result = json.load(file)
            d = []
            key = root.split('/')[-1].split('_')[0]
            if key.isnumeric():
                key = int(key)
            else:
                key = 0
            d.append(key)
            d.append(result["bop19_average_recall"])
            d.append(result["bop19_average_recall_mspd"])
            d.append(result["bop19_average_recall_mssd"])
            d.append(result["bop19_average_recall_vsd"])
            d.append(result["bop19_average_time_per_image"])
            data.append(d)
            count += 1
data.sort(key=lambda x: x[0], reverse=False)
# print(count)
# print(data)

df = pd.DataFrame(data, columns=['obj', 'recall', 'mspd', 'mssd', 'vsd', 'time'], dtype=float)
df.to_csv(os.path.join(dir_path, 'result.csv'))
