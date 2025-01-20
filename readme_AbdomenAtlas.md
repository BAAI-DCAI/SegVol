## Readme for [AbdomenAtlas](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini)

### Training setting
**Hyper-parameters**
* patch size: (4, 16, 16)
* batch size: 4 per GPU(8x GPU in total)
* optimizer: AdamW
* learning rate: 1e-4
* loss function: BCE loss+Dice loss
* weight decay: 1e-5
* param 181.0M 
* inference time 0.52 *µs/mm3*
* inference memory 0.8 GB

**Dataset:**

Prior to training on AbdomenAtlas, SegVol is pre-trained on 90K unlabeled Computed Tomography (CT) volumes from M3D-Cap, and 5,772 labeled CT volumes from M3D-Seg.
After that, SegVol is fine-tuning on AbdomenAtlas with 8 x A800 GPUs.

### Model weights
Download [testing script and model weights for AbdomenAtlas1.1](https://drive.google.com/file/d/1qqUs3Jkkam4RpP7b_kv7dUdB-wrY8GcL/view?usp=drive_link).

### Model testing
Download [testing script and model weights for AbdomenAtlas1.1](https://drive.google.com/file/d/1qqUs3Jkkam4RpP7b_kv7dUdB-wrY8GcL/view?usp=drive_link) first.
#### Install requirements:

```shell
conda create -n segvol_transformers python=3.8
conda activate segvol_transformers
```

The [pytorch v1.13.1](https://pytorch.org/get-started/previous-versions/) (or higher version) is needed first. Following install key requirements using commands:

Key requirements:

```shell
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
```

Other environment dependences can refer to `requirements.txt`.

#### Configure & run test script:

Please configure the params at the head of test.py file & run it

### Citation
```
@article{du2023segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2311.13385},
  year={2023}
}
@misc{bai2024m3dadvancing3dmedical,
      title={M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models}, 
      author={Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao},
      year={2024},
      eprint={2404.00578},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.00578}, 
}
@misc{bassi2024touchstonebenchmarkrightway,
      title={Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?}, 
      author={Pedro R. A. S. Bassi and Wenxuan Li and Yucheng Tang and Fabian Isensee and Zifu Wang and Jieneng Chen and Yu-Cheng Chou and Yannick Kirchhoff and Maximilian Rokuss and Ziyan Huang and Jin Ye and Junjun He and Tassilo Wald and Constantin Ulrich and Michael Baumgartner and Saikat Roy and Klaus H. Maier-Hein and Paul Jaeger and Yiwen Ye and Yutong Xie and Jianpeng Zhang and Ziyang Chen and Yong Xia and Zhaohu Xing and Lei Zhu and Yousef Sadegheih and Afshin Bozorgpour and Pratibha Kumari and Reza Azad and Dorit Merhof and Pengcheng Shi and Ting Ma and Yuxin Du and Fan Bai and Tiejun Huang and Bo Zhao and Haonan Wang and Xiaomeng Li and Hanxue Gu and Haoyu Dong and Jichen Yang and Maciej A. Mazurowski and Saumya Gupta and Linshan Wu and Jiaxin Zhuang and Hao Chen and Holger Roth and Daguang Xu and Matthew B. Blaschko and Sergio Decherchi and Andrea Cavalli and Alan L. Yuille and Zongwei Zhou},
      year={2024},
      eprint={2411.03670},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.03670}, 
}
```
