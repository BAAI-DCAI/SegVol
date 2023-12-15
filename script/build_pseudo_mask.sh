# ct & gt should be generated first
export datasets_root="path/to/dataset_post"
export dataset_code="0011"

CUDA_VISIBLE_DEVICES=0 python data_process/pseudo_mask_process.py \
-dataset_code $dataset_code \
-datasets_root $datasets_root
