import numpy as np
import os
from scipy import sparse
import ast
from felzenszwalb import _felzenszwalb_python
from monai import transforms
import multiprocessing
import argparse

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_code", type=str, required=True)
    parser.add_argument("-datasets_root", type=str, required=True)
    parser.add_argument("-spatial_size", default=(256, 256, 32), type=tuple)
    
    args = parser.parse_args()
    return args

args = set_parse()

transform = transforms.Compose(
        [
            transforms.AddChanneld(keys=["image"]),
            transforms.Resized(keys=["image", "label"], spatial_size=args.spatial_size),
        ]
    )

data_path = os.path.join(args.datasets_root, args.dataset_code, 'ct')
labels_path  = os.path.join(args.datasets_root, args.dataset_code, 'gt')
segmented_image_save_path = os.path.join(args.datasets_root, args.dataset_code, 'fh_seg')
if not os.path.exists(segmented_image_save_path):
    os.makedirs(segmented_image_save_path)

exist_file_list = os.listdir(segmented_image_save_path)
print('exist_file_list ', exist_file_list)

ct_list = sorted([item for item in os.listdir(data_path)])
gt_list = sorted([item for item in os.listdir(labels_path)])
assert len(ct_list) == len(gt_list), args.dataset_code + '---' + str(len(ct_list)) + '---' + str(len(gt_list))
process_list = [(
    f'{idx}/{len(ct_list)}',
    ct_list[idx],
    gt_list[idx],
    data_path,
    labels_path,
    segmented_image_save_path,
    ) for idx in range(len(ct_list))]
assert len(ct_list) == len(gt_list)

def read_ct_gt(ct_file_path, gt_file_path):
    # ct
    img_array = np.load(ct_file_path)[0]

    # gt
    allmatrix_sp= sparse.load_npz(gt_file_path)
    gt_shape = ast.literal_eval(gt_file_path.split('.')[-2])
    gt_array=allmatrix_sp.toarray().reshape(gt_shape)
    item = {
        "image": img_array,
        "label": gt_array,
    }
    # transform
    item_post = transform(item)
    return item_post["image"], item_post["label"]

def combine_gt_fh(gt, fh_seg):
    assert len(gt.shape) == 4
    label_idx = max(np.unique(fh_seg)) + 1
    for gt_label_idx in range(gt.shape[0]):
        fh_seg[gt[gt_label_idx]==1] = label_idx
        label_idx += 1
    return fh_seg

def run(info):
    info_str, ct_name, gt_name, data_path, labels_path, segmented_image_save_path = info
    if ct_name + '.npy' in exist_file_list:
        print(ct_name + '.npy exist, skip')
        return
    print('---> process ', info_str)
    save_file = os.path.join(segmented_image_save_path, ct_name)
    if os.path.exists(save_file):
        print(save_file, ' exist now')
        return
    ct_path = os.path.join(data_path, ct_name)
    gt_path = os.path.join(labels_path, gt_name)
    assert gt_name.split('.(')[0] == ct_name.split('.npy')[0]
    ct, gt = read_ct_gt(ct_path, gt_path)
    # felzenszwalb seg
    segmented_image = _felzenszwalb_python(ct.squeeze())
    # combine gt & felzenszwalb seg
    segmented_image = combine_gt_fh(gt, segmented_image)
    print('region ids: ', np.unique(segmented_image))
    print(f'---> save {save_file}')
    np.save(save_file, segmented_image)

with multiprocessing.Pool(processes=10) as pool:
    pool.map(run, process_list)

print('FH Pseudo Mask Build Done!')