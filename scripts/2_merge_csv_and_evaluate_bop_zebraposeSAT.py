import os
import sys
from shutil import copyfile
import glob
import pandas as pd

os.environ["BOP_PATH"] = "/home/Symnet/datasets/BOP_DATASETS"

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path


folder_path = "/home/Symnet/output/tless_pbr_only/zebraposeSAT"

found = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.startswith("zebrapose-sat_tless-test_obj") and file.endswith(".csv"):
            found.append(os.path.join(root, file))

assert len(found) == 30

combined_csv = pd.concat([pd.read_csv(f) for f in found])
output_fn = os.path.join(folder_path, "Zebrapose-sat_tless-test.csv")
combined_csv.to_csv(output_fn, index=False, encoding="utf-8-sig")


# print("python bop_toolkit/scripts/eval_bop19_pose.py " + f"--result_filenames {os.path.abspath(output_fn)} " + f"--results_path {os.path.abspath(folder_path)} " +f"--eval_path {os.path.abspath(folder_path)} " + "--targets_filename tless_test_targets_all.json")
os.system(
    "python bop_toolkit/scripts/eval_bop19_pose.py "
    + f"--result_filenames {os.path.abspath(output_fn)} "
    + f"--results_path {os.path.abspath(folder_path)} "
    + f"--eval_path {os.path.abspath(folder_path)} "
    + f">> {folder_path}/save.txt"
)
