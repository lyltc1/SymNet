import os
import sys

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))  # add project directory to sys.path


eval_folders = ['SymNet_tless_obj22_20230314_154657',
                'SymNet_tless_obj3_20230314_154641',
                'SymNet_tless_obj7_20230314_154650',
                'SymNet_tless_obj5_20230314_154647',
                'SymNet_tless_obj1_20230314_154608',
                'SymNet_tless_obj4_20230314_154751',]

for eval_folder in eval_folders:
    os.system(
        "python core/symn/run_evaluate_with_edge_refine.py --eval_folder output/" +
        eval_folder + " --debug")
