export segvol_ckpt="path/to/segvol_model_e270.pth"
export clip_dir="path/to/clip"
export work_dir="path/to/logdir"
export demo_config_path="./config/config_demo.json"

CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
--resume $segvol_ckpt \
--clip_ckpt $clip_dir \
-work_dir $work_dir \
--demo_config $demo_config_path 
