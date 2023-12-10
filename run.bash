fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file /home/SymNet/output/SymNet_ycbv_obj4_20231002_060228/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 4 
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 5
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 6
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 7
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 8
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 9
fuser -k -v /dev/nvidia*
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 10
fuser -k -v /dev/nvidia*