import os
import sys

import pandas as pd

# for boptoolkit to find correct BOP_PATH
os.environ['BOP_PATH'] = '/home/Symnet/datasets/BOP_DATASETS'

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

base_path = "output/ycbv_pbr_16bit"
output_fn = os.path.join(base_path, "YcbvkeyframeBest16bit_ycbv-test.csv")
output_dir = output_fn.split('.csv')[0]
def find_files(base_path, dir_name_start, file_name):
    paths = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root).startswith(dir_name_start) and file_name in files:
            paths.append(os.path.join(root, file_name))
    paths.sort()
    return paths

paths = []
file_name = 'SymNet_ycbv-test.csv'
dir_name_start = 'epoch'

file_paths = find_files(base_path, dir_name_start, file_name)

print(file_paths)
print("The number of folders need to be evaluated is", len(file_paths))
user_input = input("press enter to continue")
if user_input != "":
    print("did not press enter")
    exit()

combined_csv = pd.concat([pd.read_csv(f) for f in file_paths])

combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')
# ---
print("start eval add")
command = "python bop_toolkit/scripts/eval_add_for_ycbv.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}" + " --targets_filename ycbv_test_targets_keyframe.json >> " + os.path.join(output_dir, "add_result.txt")
print(command)
os.system(command)
# ---
print("start eval adi")
command = "python bop_toolkit/scripts/eval_adi_for_ycbv.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}" + " --targets_filename ycbv_test_targets_keyframe.json >> " + os.path.join(output_dir, "adi_result.txt")
print(command)
os.system(command)
# ---
print("start eval aucadd")
command = "python bop_toolkit/scripts/eval_aucadd_for_ycbv.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}" + " --targets_filename ycbv_test_targets_keyframe.json >> " + os.path.join(output_dir, "aucadd_result.txt")
print(command)
os.system(command)
# ---
print("start eval aucadi")
command = "python bop_toolkit/scripts/eval_aucadi_for_ycbv.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(base_path)} " +f"--eval_path {os.path.abspath(base_path)}" + " --targets_filename ycbv_test_targets_keyframe.json >> " +  os.path.join(output_dir, "aucadi_result.txt")
print(command)
os.system(command)