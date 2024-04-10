import os
import sys


sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path

# search all folders with property (like starting with 'xxx')
# main_folder_name = '/home/Symnet/output/symnet_ycbv_16bit/'
main_folder_name = '/home/Symnet/output/ycbv_pbr_10bit/'
eval_folders = []

for name in os.listdir(main_folder_name):
    if name.startswith('SymNet_ycbv_obj'):
        eval_folders.append(name)

# check the number of folders to be evaluated
print(eval_folders)
print("The number of folders need to be evaluated is", len(eval_folders))
user_input = input("press enter to continue")
if user_input != "":
    print("did not press enter")
    exit()

# because of comment out the evaluation code in run_evaluate.py, will only generate .csv
for eval_folder in eval_folders:
    os.system(
        "python core/symn/run_evaluate.py --eval_folder " + os.path.join(main_folder_name,
        eval_folder))
    
