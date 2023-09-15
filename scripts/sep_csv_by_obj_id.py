import os
import pandas as pd

folder = '/home/lyltc/git/SymNet/output/zebrapose'
file_name = "zebrapose_tless-test_25be093d-bc1d-43a2-bd27-afa5daefaa70.csv"

df = pd.read_csv(os.path.join(folder, file_name))
for obj_id in range(1, 31):
    new_data = df.loc[df['obj_id'] == obj_id]
    result_filename = "zebrapose_tless-test_obj"+str(obj_id)+".csv"
    new_data.to_csv(os.path.join(folder, result_filename), index=False)
    os.system(
        f'python /home/lyltc/git/GDR-Net/bop_toolkit/scripts/eval_bop19_pose.py \
        --result_filenames {result_filename} \
        --results_path /home/lyltc/git/SymNet/output/zebrapose \
        --eval_path /home/lyltc/git/SymNet/output/zebrapose')


