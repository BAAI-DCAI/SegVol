import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torch
import monai.transforms as transforms

def draw_result(category, image, bboxes, points, logits, gt3D, spatial_size, work_dir):
    zoom_out_transform = transforms.Compose([
        transforms.AddChanneld(keys=["image", "label", "logits"]),
        transforms.Resized(keys=["image", "label", "logits"], spatial_size=spatial_size, mode='nearest-exact')
    ])
    post_item = zoom_out_transform({
        'image': image,
        'label': gt3D,
        'logits': logits
    })
    image, gt3D, logits = post_item['image'][0], post_item['label'][0], post_item['logits'][0]
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).int()

    root_dir=os.path.join(work_dir, f'fig_examples/{category}/') 

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if bboxes is not None:
        x1, y1, z1, x2, y2, z2 = bboxes[0].cpu().numpy()
    if points is not None:
        points = (points[0].cpu().numpy(), points[1].cpu().numpy())
        points_ax = points[0][0]   # [n, 3]
        points_label = points[1][0] # [n]

    for j in range(image.shape[0]):
        img_2d = image[j, :, :].detach().cpu().numpy()
        preds_2d = preds[j, :, :].detach().cpu().numpy()
        label_2d = gt3D[j, :, :].detach().cpu().numpy()
        if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
            continue

        img_2d = img_2d * 255
        # orginal img
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img_2d, cmap='gray')
        ax1.set_title('Image with prompt') 
        ax1.axis('off')

        # gt
        ax2.imshow(img_2d, cmap='gray')
        show_mask(label_2d, ax2)
        ax2.set_title('Ground truth') 
        ax2.axis('off')

        # preds
        ax3.imshow(img_2d, cmap='gray')
        show_mask(preds_2d, ax3)
        ax3.set_title('Prediction') 
        ax3.axis('off')

        # boxes
        if bboxes is not None:
            if j >= x1 and j <= x2:
                show_box((z1, y1, z2, y2), ax1)
        # points
        if points is not None:
            for point_idx in range(points_label.shape[0]):
                point = points_ax[point_idx]
                label = points_label[point_idx] # [1]
                if j == point[0]:
                    show_points(point, label, ax1)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(root_dir, f'{category}_{j}.png'), bbox_inches='tight')
        plt.close()

def show_mask(mask, ax):
    color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.35)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def show_points(points_ax, points_label, ax):
    color = 'red' if points_label == 0 else 'blue'
    ax.scatter(points_ax[2], points_ax[1], c=color, marker='o', s=200)