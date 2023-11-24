# SegVol: Universal and Interactive Volumetric Medical Image Segmentation
This repo is the official implementation of [SegVol: Universal and Interactive Volumetric Medical Image Segmentation](https://arxiv.org/abs/2311.13385).

## News
(2023.11.24) *You can download weight files of SegVol and ViT(CTs pre-train) [here](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link).*

(2023.11.23) *The brief introduction and instruction are comming soon in this week.*

(2023.11.23) *The inference demo code has been uploaded.*

(2023.11.22) *The first edition of our paper has been uploaded to arXiv.*

## Introduction
The SegVol is a universal and interactive model for volumetric medical image segmentation. SegVol accepts **point**, **box** and **text** prompt while output volumetric segmentation. By training on 90k unlabeled Computed Tomography (CT) volumes and 6k labeled CTs, this foundation model supports the segmentation of over 200 anatomical categories.

We will release SegVol's **inference code**, **training code**, **model params** and **ViT pre-training params** (pre-training is performed over 2,000 epochs on 96k  CTs). 

## Usage
### Requirements
The [pytorch v1.11.0](https://pytorch.org/get-started/previous-versions/) (or higher virsion) is needed first. Following install MONAI, einops and transformers using commands:

```
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
``` 
### Config and run demo script
Please download the demo dataset: [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K). Set CT path and Ground Truth path of a case in the [config_demo.json](https://github.com/BAAI-DCAI/SegVol/blob/main/config/config_demo.json).

After that, config the [inference_demo.sh](https://github.com/BAAI-DCAI/SegVol/blob/main/script/inference_demo.sh) file for execution:

- `$segvol_ckpt`: the path of SegVol's checkpoint (Download from [here](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link)).

- `$work_dir`: any path of folder you want to save the log files and visualizaion results.

Finally, you can control the prompt type, zoom-in-zoom-out mechanism and visualizaion process at [here](https://github.com/BAAI-DCAI/SegVol/blob/35f3ff9c943a74f630e6948051a1fe21aaba91bc/inference_demo.py#L208C11-L208C11).

Now, just run `bash script/inference_demo.sh` to infer your demo case.

## Citation
If you find this repository helpful, please consider citing:
```
@misc{du2023segvol,
      title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation}, 
      author={Yuxin Du and Fan Bai and Tiejun Huang and Bo Zhao},
      year={2023},
      eprint={2311.13385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Thanks for the following amazing works:

[HuggingFace](https://huggingface.co/).

[CLIP](https://github.com/openai/CLIP).

[MONAI](https://github.com/Project-MONAI/MONAI).

[Image by brgfx](https://www.freepik.com/free-vector/anatomical-structure-human-bodies_26353260.htm) on Freepik.

[Image by muammark](https://www.freepik.com/free-vector/people-icon-collection_1157380.htm#query=user&position=2&from_view=search&track=sph) on Freepik.

[Image by pch.vector](https://www.freepik.com/free-vector/different-phone-hand-gestures-set_9649376.htm#query=Vector%20touch%20screen%20hand%20gestures&position=4&from_view=search&track=ais) on Freepik.

[Image by starline](https://www.freepik.com/free-vector/set-three-light-bulb-represent-effective-business-idea-concept_37588597.htm#query=idea&position=0&from_view=search&track=sph) on Freepik.




