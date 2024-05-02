""" Evaluate BOP for SymNet-ZebraCode under folder output/SymNet_pbr_ZebraCode """

import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

folder_path = "SymNet_pbr_ZebraCode"
dir_suffix = "zebracode"
output_fn = os.path.join(folder_path, "SymNetZebraCodepbr_tless-test.csv")
RUN_EVALUATE = False
GENERATE_CSV = False
EVAL_BOP = False

if RUN_EVALUATE:
    eval_folders = []
    for name in os.listdir(folder_path):
        if name.startswith("SymNet_tless_obj"):
            eval_folders.append(name)
    # check the number of folders to be evaluated
    assert len(eval_folders) == 30, f"number of folder wrong, {len(eval_folders)} != 30"

    for eval_folder in eval_folders:
        command = f"python core/symn/run_evaluate.py --eval_folder {os.path.join(folder_path, eval_folder)} --ckpt best --detection zebrapose_detections/tless/tless_bop_pbr_only.json --dir_suffix {dir_suffix}"
        if os.system(command) != 0:
            print(f"Command failed, {command}")
            sys.exit(1)

if GENERATE_CSV:
    # find all the file "SymNet_tless-test.csv" whose parent folder end with suffix 'zebracode'
    eval_files = []

    def find_files(start_path, filename, dir_suffix):
        for root, dirs, files in os.walk(start_path):
            if root.endswith(dir_suffix) and filename in files:
                eval_files.append(os.path.join(root, filename))

    find_files(folder_path, "SymNet_tless-test.csv", dir_suffix)
    print(eval_files)
    assert len(eval_files) == 30

    combined_csv = pd.concat([pd.read_csv(f) for f in eval_files])
    combined_csv.to_csv(output_fn, index=False, encoding="utf-8-sig")

if EVAL_BOP:
    command = f"python bop_toolkit/scripts/eval_bop19_pose.py --result_filenames {output_fn} --results_path {os.path.abspath(folder_path)} --eval_path {os.path.abspath(folder_path)}"
    print(command)
    os.system(command)
