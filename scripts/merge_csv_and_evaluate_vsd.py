""" copy all the csv file and generate SymNetPbr_tless-test.csv """

import os
import sys
from shutil import copyfile
import glob
import pandas as pd
os.environ['BOP_PATH'] = '/home/Symnet/datasets/BOP_DATASETS'

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path


folder_path = "/home/Symnet/output/tless_all_primesense_results/pbr_trained"

# found = list()
# # 遍历文件夹
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         file_path = os.path.join(root, file)
#         if "SymNet_tless-test.csv" in file_path and "result_all" in file_path and "epoch=" in file_path:
#             obj_str = file_path.split("obj")[1].split("_")[0]
#             found.append(obj_str)
#             target_path = os.path.join(folder_path, obj_str + "_tless-test.csv")
#             copyfile(file_path, target_path)
# found.sort()
# print(found)

# extension = 'csv'
# all_filenames = [i for i in glob.glob(folder_path+'/*.{}'.format(extension), recursive=False)]
# print(all_filenames)

# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
output_fn = os.path.join(folder_path, "SymNetPbr_tless-test.csv")
# combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')


# print("python bop_toolkit/scripts/eval_vsd_tau_20_threshold_0.3.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(folder_path)} " +f"--eval_path {os.path.abspath(folder_path)} " + "--targets_filename tless_test_targets_all.json")
os.system("python bop_toolkit/scripts/eval_vsd_tau_20_threshold_0.3.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(folder_path)} " +f"--eval_path {os.path.abspath(folder_path)} " + "--targets_filename tless_test_targets_all.json")