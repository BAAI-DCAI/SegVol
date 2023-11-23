export segvol_ckpt="/zhaobai46a01/code/MedSAM-main/work_dir/MedSAM-ViT-B-20231103-0933/medsam_model_e270.pth"
export clip_dir="/zhaobai46a01/pretrain_mods/clip"
export work_dir="/zhaobai46a01/code/SegVol/work_dir"
export demo_config_path="./config/config_demo.json"

CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
--resume $segvol_ckpt \
--clip_ckpt $clip_dir \
-work_dir $work_dir \
--demo_config $demo_config_path 
