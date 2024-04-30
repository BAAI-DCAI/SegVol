import numpy as np
import os
from scipy import sparse
import ast
from felzenszwalb import _felzenszwalb_python
from monai import transforms
import multiprocessing
import argparse
join = os.path.join

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
case_dir = join(args.datasets_root, args.dataset_code)
case_list = sorted([item for item in os.listdir(case_dir) if '.json' not in item])

process_list = [(
    f'{idx}/{len(case_list)}',
    case_list[idx],
    case_dir,
    ) for idx in range(len(case_list))]

def read_ct_gt(ct_file_path, gt_file_path):
    # ct
    img_array = np.load(ct_file_path)[0]

    # gt
    allmatrix_sp= sparse.load_npz(gt_file_path)
    gt_shape = ast.literal_eval(gt_file_path.split('.')[-2].split('_')[-1])
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
    info_str, case_name, case_dir_path = info
    save_file = join(case_dir_path, case_name, 'pseudo_mask.npy')

    if os.path.exists(save_file):
        print(case_name + ' pseudo_mask.npy exist, skip')
        return
    print('---> process ', info_str)

    case_files = sorted(os.listdir(join(case_dir_path, case_name)))
    ct_path = join(case_dir_path, case_name, case_files[0])
    gt_path = join(case_dir_path, case_name, case_files[1])

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