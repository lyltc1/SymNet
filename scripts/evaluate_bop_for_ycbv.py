""" the folders in output which starts with "Symnet_ycbv_obj*, find all the SymNet_ycbv-test.csv under folder "last" and merge and evaluation
"""

import os
import sys
from shutil import copyfile
import glob
import pandas as pd
os.environ['BOP_PATH'] = '/home/Symnet/datasets/BOP_DATASETS'

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

def find_files(base_path, dir_name, file_name):
    paths = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == dir_name and file_name in files:
            paths.append(os.path.join(root, file_name))
    paths.sort()
    return paths

paths = []
base_path = "/home/Symnet/output/symnet_ycbv_16bit"
dir_name = 'last'
file_name = 'SymNet_ycbv-test.csv'
file_paths = find_files(base_path, dir_name, file_name)

for path in file_paths:
    print(path)

assert len(file_paths) == 21

combined_csv = pd.concat([pd.read_csv(f) for f in file_paths])
output_fn = os.path.join(base_path, "SymNetpbr_ycbv-test.csv")
combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')


os.system("python bop_toolkit/scripts/eval_bop19_pose.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}")