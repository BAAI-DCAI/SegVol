import numpy as np
import monai.transforms as transforms


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d

class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d

class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d
    
    def normalize(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray
    

def process_ct_gt(case_path, label_path, category, spatial_size):
    print('Data preprocessing...')
    # transform
    img_loader = transforms.LoadImage()
    transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            ForegroundNormalization(keys=["image"]),
            DimTranspose(keys=["image", "label"]),
            MinMaxNormalization(),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, mode='constant'),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    zoom_out_transform = transforms.Resized(keys=["image", "label"], spatial_size=spatial_size, mode='nearest-exact')

    ###
    item = {}
    # generate ct_voxel_ndarray
    ct_voxel_ndarray, _ = img_loader(case_path)
    print(type(ct_voxel_ndarray))
    ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
    ct_shape = ct_voxel_ndarray.shape
    ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
    item['image'] = ct_voxel_ndarray

    # generate gt_voxel_ndarray
    gt_voxel_ndarray, _ = img_loader(label_path)
    gt_voxel_ndarray = np.array(gt_voxel_ndarray)
    present_categories = np.unique(gt_voxel_ndarray)
    gt_masks = []
    for cls_idx in range(len(category)):
        # ignore background
        cls = cls_idx + 1
        if cls not in present_categories:
            gt_voxel_ndarray_category = np.zeros(ct_shape)
            gt_masks.append(gt_voxel_ndarray_category)
        else:
            gt_voxel_ndarray_category = gt_voxel_ndarray.copy()
            gt_voxel_ndarray_category[gt_voxel_ndarray != cls] = 0
            gt_voxel_ndarray_category[gt_voxel_ndarray == cls] = 1
            gt_masks.append(gt_voxel_ndarray_category)
    gt_voxel_ndarray = np.stack(gt_masks, axis=0)
    assert gt_voxel_ndarray.shape[0] == len(category) and gt_voxel_ndarray.shape[1:] == ct_voxel_ndarray.shape[1:]
    item['label'] = gt_voxel_ndarray.astype(np.int32)

    # transform
    item = transform(item)
    item_zoom_out = zoom_out_transform(item)
    item['zoom_out_image'] = item_zoom_out['image']
    item['zoom_out_label'] = item_zoom_out['label']
    print(  'Zoom_in image shape: ',   item['image'].shape, 
          '\nZoom_in label shape: ', item['label'].shape,
          '\nZoom_out image shape: ', item['zoom_out_image'].shape,
          '\nZoom_out label shape: ', item['zoom_out_label'].shape,
          )
    return item
