""" Specifically run two ckpt for icbin """

import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

folder_path = "output/icbin_pbr"
output_fn = os.path.join(folder_path, "SymNet_icbin-test.csv")
RUN_EVALUATE = False
GENERATE_CSV = False
EVAL_BOP = True

if RUN_EVALUATE:
    command = "python core/symn/run_evaluate.py --eval_folder output/icbin_pbr/SymNet_icbin_obj2_20240411_232928/ --ckpt epoch=40-step=537838.ckpt"
    os.system(command)

    command = "python core/symn/run_evaluate.py --eval_folder output/icbin_pbr/SymNet_icbin_obj1_20240410_130645/ --ckpt epoch=39-step=468280.ckpt"
    os.system(command)

if GENERATE_CSV:
    found = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if (
                "epoch=39-step=468280" in root or "epoch=40-step=537838" in root
            ) and file.endswith(".csv"):
                found.append(os.path.join(root, file))

    print(found)
    assert len(found) == 2

    combined_csv = pd.concat([pd.read_csv(f) for f in found])
    combined_csv.to_csv(output_fn, index=False, encoding="utf-8-sig")

if EVAL_BOP:
    command = f"python bop_toolkit/scripts/eval_bop19_pose.py --result_filenames SymNet_icbin-test.csv --results_path {os.path.abspath(folder_path)} --eval_path {os.path.abspath(folder_path)}"
    print(command)
    os.system(command)
