### Guideline for training SegVol

#### Build universal datasets
📂 If the 25 processed datasets in our [ModelScope/魔搭社区](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary) or [HuggingFace](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary) have been downloaded, you can **skip** this step.

1. We use the [Abdomenct-12organ](https://zenodo.org/records/7860267) as demo dataset. 
2. After downloading the demo dataset, you need to config the [script/build_dataset.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/build_dataset.sh) file to set the environment vars:
    * `$SAVE_ROOT` is the save path for the post-processed datasets.
    * `$DATASET_CODE`  is your custom id for your dataset. We suggest you use  `0000`, `0001`, ... as the dataset  id.
    * `$IMAGE_DIR` and `$LABEL_DIR` is the image directory path and label directory path of the original demo dataset.
    * `$TEST_RATIO` is the ratio of preserved val/test data from the whole set.
3. **Set the `category` in [data_process/train_data_process.py](https://github.com/BAAI-DCAI/SegVol/blob/95e3f5f3c62b68fa63dbccb011a4c657642e1445/data_process/train_data_process.py#L19C1-L19C1) file.** Categories should be in the same order as the corresponding idx in ground truth volume and `background` category should be ignored.
4. Just run `bash script/build_dataset.sh`.

*If you want to combine **multiple datasets**, you can run the [script/build_dataset.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/build_dataset.sh) for multiple times and assign different `$DATASET_CODE` for each dataset.*

#### Build pseudo mask labels

After the process of building universal datasets finished, you should build pseudo mask labels for each CT in the post-processed datasets.

1. You will need to config the [script/build_pseudo_mask.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/build_pseudo_mask.sh) first:
    * `$DATASET_ROOT` is the directory path for the post-processed datasets.
    * `$DATASET_CODE` is the custom code of your post-processed dataset.
2. Run `bash script/build_pseudo_mask.sh`. The pseudo masks for the `$DATASET_CODE` dataset will be generated at `$DATASET_ROOT/$DATASET_CODE/fh_seg`.

*If you combine **multiple datasets**, you should run the [script/build_pseudo_mask.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/build_dataset.sh) for each dataset.*

#### Training

1. Make sure you have completed the above steps correctly.
2. Set environment vars in [script/train.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/train.sh):
    * `$SEGVOL_CKPT` is the weight file of SegVol.(Download from [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main)🤗 or [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link))
    * `$WORK_DIR` is save path for log files and checkpoint files in the training phase.
    * `$DATA_DIR` is the directory path for the above post-processed datasets.
    * Define [dataset_codes](https://github.com/BAAI-DCAI/SegVol/blob/95e3f5f3c62b68fa63dbccb011a4c657642e1445/train.py#L22C32-L22C32) to indicate which datasets are used for training
    * Configure [these parameters](https://github.com/BAAI-DCAI/SegVol/blob/95e3f5f3c62b68fa63dbccb011a4c657642e1445/train.py#L16) according to your training needs.
    * Set the `$CUDA_VISIBLE_DEVICES` according to  your devices.
3. Run `bash script/train.sh`.

#### Training from scratch

If you want to training from scratch without our SegVol checkpoint, I highly recommend that you use the pre-trained ViT [here](https://github.com/BAAI-DCAI/SegVol/blob/95e3f5f3c62b68fa63dbccb011a4c657642e1445/train.py#L148C4-L148C4) and modify [here](https://github.com/BAAI-DCAI/SegVol/blob/95e3f5f3c62b68fa63dbccb011a4c657642e1445/network/model.py#L206C20-L206C20) to load the CLIP TextEncoder parameters.

