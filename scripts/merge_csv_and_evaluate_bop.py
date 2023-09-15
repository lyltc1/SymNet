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


folder_path = "/home/Symnet/output/tless_real_pbr_bop_results/"

extension = 'csv'
all_filenames = [i for i in glob.glob(folder_path+'/*.{}'.format(extension), recursive=False)]
print(len(all_filenames))

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
output_fn = os.path.join(folder_path, "SymNetPbr_tless-test.csv")
combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')


os.system("python bop_toolkit/scripts/eval_bop19_pose.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(folder_path)} " +f"--eval_path {os.path.abspath(folder_path)}")