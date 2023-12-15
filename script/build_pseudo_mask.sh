# ct & gt should be generated first
export DATASET_ROOT="path/to/dataset_post"
export DATASET_CODE="0000"

python data_process/pseudo_mask_process.py \
-dataset_code $DATASET_CODE \
-datasets_root $DATASET_ROOT
