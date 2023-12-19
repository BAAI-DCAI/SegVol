### Using our pre-trained ViT as your model encoder

![pretrainloss](https://github.com/BAAI-DCAI/SegVol/assets/60123629/88707c9d-b4fc-4675-90da-9d3dc24de886)

We pre-train ViT on 96k CTs for over 2,000 epochs. The pre-trained ViT shows excellent generalization performance and the ability to accelerate convergence. 
A simple experiment is performed on [AMOS22](https://amos22.grand-challenge.org/), training [UNETR](https://arxiv.org/abs/2103.10504) with and without pre-trained encoder:

|   Model  | Encoder       |   Dice score(%)   |
| :--:     | :--:          |:--:               |
|   UNETR  | w/o pre-train |  67.12            |
|   UNETR  | w   pretrain  |  79.10            |


You can use the ViT independently as your model's encoder. The pre-trained ViT weight file is uploaded at [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main)🤗 and [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link). The demo code is as follows:
```python
import torch
from monai.networks.nets import ViT

vit_checkpoint = 'path/to/ViT_pretrain.ckpt'

vit = ViT(
        in_channels=1,
        img_size=(32,256,256),
        patch_size=(4,16,16),
        pos_embed="perceptron",
        )
print(vit)

with open(vit_checkpoint, "rb") as f:
    state_dict = torch.load(f, map_location='cpu')['state_dict']
    encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
vit.load_state_dict(encoder_dict)
print(f'Image_encoder load param: {vit_checkpoint}')
```
