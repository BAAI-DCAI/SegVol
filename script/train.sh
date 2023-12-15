export SEGVOL_CKPT="path/to/SegVol_v1.pth"
export WORK_DIR="./work_dir"
export DATA_DIR="path/to/dataset_post"
# !!! remember to config the dataset_codes used to train in train.py !!!

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--resume $SEGVOL_CKPT \
-work_dir $WORK_DIR \
--data_dir $DATA_DIR
