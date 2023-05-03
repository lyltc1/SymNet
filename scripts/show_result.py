import os
import sys
import json


sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

f_result = {}
count = 0
sum_ = 0

for name in os.listdir('/home/lyltc/git/SymNet/output'):
    if '2023040' in name:
        f = os.path.join(os.getcwd(), 'output', name, 'result_bop', 'SymNet_tless-test', 'scores_bop19.json')
        with open(f, 'r') as file:
            result = json.load(file)
            bop19_recall = result['bop19_average_recall']
            sum_ += bop19_recall
            obj_id = int(name.split('obj')[1].split('_')[0])
            f_result[obj_id] = bop19_recall
print(f_result)
print(sum_)