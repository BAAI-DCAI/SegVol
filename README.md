# SegVol: Universal and Interactive Volumetric Medical Image Segmentation
<div align="center">
  
  <img src="https://github.com/BAAI-DCAI/SegVol/assets/60123629/6f56fc35-1d75-468c-ab82-1e0cf47eb83e" width="85%" height="85%">
  
 | 🌟**Quickstart([ModelScope](https://www.modelscope.cn/models/yuxindu/SegVol/summary) / [🤗HF](https://huggingface.co/BAAI/SegVol))** | 📃 [**Paper**](https://arxiv.org/abs/2311.13385) | [**Web Tool**](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary) | 📂 **Datasets([ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary)/[🤗HF](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg))** |
</div>

The SegVol is a universal and interactive model for volumetric medical image segmentation. SegVol accepts **point**, **box** and **text** prompt while output volumetric segmentation. By training on 90k unlabeled Computed Tomography (CT) volumes and 6k labeled CTs, this foundation model supports the segmentation of over 200 anatomical categories.

We have released SegVol's **inference code**, **training code**, **model params** and **ViT pre-training params** (pre-training is performed over 2,000 epochs on 96k  CTs). 

**Keywords**: 3D medical SAM, volumetric image segmentation

## Quickstart: Enable easy training and testing
### 🌟[Quickstart](https://www.modelscope.cn/models/yuxindu/SegVol/summary) with ModelScope (无需代理)
### 🌟[Quickstart](https://huggingface.co/BAAI/SegVol) with HuggingFace

## Start with source code
### Requirements
The [pytorch v1.11.0](https://pytorch.org/get-started/previous-versions/) (or a higher version) is needed first. Following install key requirements using commands:

```
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
pip install matplotlib
```

### Guideline for training and inference
[How to infer a demo case](https://github.com/BAAI-DCAI/SegVol/blob/main/documents/inference_demo.md).

[How to train SegVol](https://github.com/BAAI-DCAI/SegVol/blob/main/documents/training.md).

[How to use our pre-trained ViT as your model encoder](https://github.com/BAAI-DCAI/SegVol/blob/main/documents/pretrained_vit.md).

### Datasets involved
🌟The 25 processed datasets are being uploaded to [ModelScope/魔搭社区](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary) and [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg).


Links to the original datasets:
| Dataset  | Link |
| ------------- | ------------- |
| 3D-IRCADB  | https://www.kaggle.com/datasets/nguyenhoainam27/3dircadb |
|AbdomenCT-1k|	https://github.com/JunMa11/AbdomenCT-1K|
|AMOS22|	https://amos22.grand-challenge.org/|
|BTCV|	https://www.synapse.org/\#!Synapse:syn3193805/wiki/217752|
|CHAOS|	https://chaos.grand-challenge.org/|
|CT-ORG|	https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890|
|FLARE22|	https://flare22.grand-challenge.org/|
|HaN-Seg|	https://han-seg2023.grand-challenge.org/|
|KiPA22|	https://kipa22.grand-challenge.org/|
|KiTS19|	https://kits19.grand-challenge.org/|
|KiTS23|	https://kits-challenge.org/kits23/|
|LUNA16|	https://luna16.grand-challenge.org/Data/|
|MSD-Colon|	http://medicaldecathlon.com/|
|MSD-HepaticVessel|	http://medicaldecathlon.com/|
|MSD-Liver|	http://medicaldecathlon.com/|
|MSD-lung|  	http://medicaldecathlon.com/|
|MSD-pancreas|	http://medicaldecathlon.com/|
|MSD-spleen|	http://medicaldecathlon.com/|
|Pancreas-CT|	https://wiki.cancerimagingarchive.net/display/public/pancreas-ct|
|QUBIQ|	https://qubiq.grand-challenge.org/|
|SLIVER07|	https://sliver07.grand-challenge.org/|
|TotalSegmentator|	https://github.com/wasserth/TotalSegmentator|
|ULS23|	https://uls23.grand-challenge.org/|
|VerSe19|	https://osf.io/nqjyw/|
|VerSe20|	https://osf.io/t98fz/|
|WORD|	https://paperswithcode.com/dataset/word|

## [Web Tool](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary) of SegVol 📽
https://github.com/BAAI-DCAI/SegVol/assets/60123629/242a1578-e418-463c-9d53-a62eeb154c7d

## 🏆Internal Validation Performance(Dice Score)
<div align="center">

  ![github(7)](https://github.com/BAAI-DCAI/SegVol/assets/60123629/a578a66a-ddef-457a-8bf7-9ca5c8a9ba1c)

</div>

<span id="jump"></span>

## 🏆External Validation Performance(Dice Score)
<div align="center">
  <img src="https://github.com/BAAI-DCAI/SegVol/assets/60123629/2f3b4683-f4c3-4f61-b108-f21d80ba5904" width="75%" height="75%">
  
  ![github(9)](https://github.com/BAAI-DCAI/SegVol/assets/60123629/7dac6593-f1c7-4dbf-b5b8-d9f6bdf7b3ae)

</div>



We performed an external validation experiment using a novel annotated dataset from the [ULS23 Challenge](https://uls23.grand-challenge.org/) (750 + 744 + 124 cases about lesions) and the validation dataset from [Amos22](https://amos22.grand-challenge.org/) (120 cases about organs). SegVol showed strong segmentation abilities compared to other medical SAM methods in accurately segmenting lesions and 15 important organs.

### Visualization🔍

#### Dataset (Released)
![页-2](https://github.com/BAAI-DCAI/SegVol/assets/60123629/5a26a956-0112-4d22-b351-921555772887)


#### Internal Validation
![页-1](https://github.com/BAAI-DCAI/SegVol/assets/60123629/9ca9467e-e916-4116-bb0f-68eea7655ea0)


#### External Validation
![vis](https://github.com/BAAI-DCAI/SegVol/assets/60123629/d70098ec-d8cf-4b16-8b2b-4233cb992720)




## News🚀
(2024.01.03) *A radar map about [**zero-shot experiment**](#jump) has been reported.* 🏆

(2023.12.25) *Our web tool **supports download results** now! You can use it as an online tool.* 🔥🔥🔥

(2023.12.15) *The training code has been uploaded!*

(2023.12.04) ***A web tool of SegVol is [here](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary)! Just enjoy it!*** 🔥🔥🔥

(2023.11.28) *Our model and demo case have been open-source at [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main).* 🤗🤗

(2023.11.28) *The usage of pre-trained ViT has been uploaded.* 

(2023.11.24) *You can download weight files of SegVol and ViT(CTs pre-train) from [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link).* 🔥🔥🔥

(2023.11.23) *The brief introduction and instruction have been uploaded.*

(2023.11.23) *The inference demo code has been uploaded.*

(2023.11.22) *The first edition of our paper has been uploaded to arXiv.* 📃

## Citation
If you find this repository helpful, please consider citing:
```
@article{du2023segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2311.13385},
  year={2023}
}
```

## Acknowledgement
Thanks for the following amazing works:

[HuggingFace](https://huggingface.co/).

[CLIP](https://github.com/openai/CLIP).

[MONAI](https://github.com/Project-MONAI/MONAI).

[3D Slicer](https://www.slicer.org/).

[Image by brgfx](https://www.freepik.com/free-vector/anatomical-structure-human-bodies_26353260.htm) on Freepik.

[Image by muammark](https://www.freepik.com/free-vector/people-icon-collection_1157380.htm#query=user&position=2&from_view=search&track=sph) on Freepik.




