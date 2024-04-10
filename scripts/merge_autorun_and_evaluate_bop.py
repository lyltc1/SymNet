""" autorun_generate_csv_for_folder.py will generate .csv file for each folders
    this script will find all the csv file (best or last) and merge and evaluate.
"""
import os
import sys

import pandas as pd

# for boptoolkit to find correct BOP_PATH
os.environ['BOP_PATH'] = '/home/Symnet/datasets/BOP_DATASETS'

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

def find_files(base_path, dir_name_start, file_name):
    paths = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root).startswith(dir_name_start) and file_name in files:
            paths.append(os.path.join(root, file_name))
    paths.sort()
    return paths

paths = []
base_path = "output/ycbv_pbr_10bit"
dir_name_start = 'epoch'
file_name = 'SymNet_ycbv-test.csv'
file_paths = find_files(base_path, dir_name_start, file_name)

print(file_paths)
print("The number of folders need to be evaluated is", len(file_paths))
user_input = input("press enter to continue")
if user_input != "":
    print("did not press enter")
    exit()

combined_csv = pd.concat([pd.read_csv(f) for f in file_paths])
output_fn = os.path.join(base_path, "SymNetPbr"+ dir_name_start + "_ycbv-test.csv")
combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')


os.system("python bop_toolkit/scripts/eval_bop19_pose.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}")