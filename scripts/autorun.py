import os
import sys


sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path


eval_folders = []
count = 0

for name in os.listdir('/home/Symnet/output/tless_new_trained_models'):
    eval_folders.append(name)
print(eval_folders)
input("press enter to continue")
for eval_folder in eval_folders:
    os.system(
        "python core/symn/run_evaluate.py --eval_folder output/tless_new_trained_models/" +
        eval_folder)
    