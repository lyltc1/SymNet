import os
import sys
import shutil
import pandas as pd

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

main_folder_name = "/home/Symnet/output/tless_pbr_only"
DELETE_OLD_RESULT = True
GENERATE_RESULT = True
EVAL_BOP_RESULT = True


def delete_dirs(start_path, dir_names):
    for root, dirs, files in os.walk(start_path):
        for dir in dirs:
            if dir in dir_names:
                full_path = os.path.join(root, dir)
                shutil.rmtree(full_path)
                print(f"Deleted: {full_path}")


if DELETE_OLD_RESULT:
    delete_dirs(main_folder_name, ["lightning_logs", "result_bop", "visualize"])

if GENERATE_RESULT:
    # search all folders with property (like starting with 'xxx')
    eval_folders = []
    for name in os.listdir(main_folder_name):
        if name.startswith("SymNet_tless_obj"):
            eval_folders.append(name)
    # check the number of folders to be evaluated
    assert len(eval_folders) == 30, f"number of folder wrong, {len(eval_folders)} != 30"

    for eval_folder in eval_folders:
        command = f"python core/symn/run_evaluate.py --eval_folder {os.path.join(main_folder_name, eval_folder)} --detection gdrnppdet-pbr/gdrnppdet-pbr_tless-test_bed88a8e-1e0e-405b-8c62-8e7b83cf8934.json"
        if os.system(command) != 0:
            print(f"Command failed, {command}")
            sys.exit(1)


if EVAL_BOP_RESULT:
    # find all the file "SymNet_tless-test.csv" whose parent folder not end with suffix 'ablation_pnp'
    eval_files = []

    def find_files(start_path, filename, dir_suffix):
        for root, dirs, files in os.walk(start_path):
            if root.endswith(dir_suffix) and filename in files:
                eval_files.append(os.path.join(root, filename))

    find_files(main_folder_name, "SymNet_tless-test.csv", "evalgdrdet")

    eval_files.sort()
    assert len(eval_files) == 30
    # Print full paths
    combined_csv = pd.concat([pd.read_csv(f) for f in eval_files])
    output_fn = os.path.join(main_folder_name, "SymNetPbr_tless-test.csv")
    combined_csv.to_csv(output_fn, index=False, encoding="utf-8-sig")

    command = f"python bop_toolkit/scripts/eval_bop19_pose.py --result_filenames {output_fn} --results_path {os.path.abspath(main_folder_name)} --eval_path {os.path.abspath(main_folder_name)}"
    print(command)
    os.system(command)
