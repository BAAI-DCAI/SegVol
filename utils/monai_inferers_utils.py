# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import random

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

tqdm, _ = optional_import("tqdm", name="tqdm")

__all__ = ["sliding_window_inference"]

def logits2roi_coor(spatial_size, logits_global_single):
    # crop predict
    pred_global_single = torch.sigmoid(logits_global_single) > 0.5
    ## get all pos idx
    nonzero_indices = torch.nonzero(pred_global_single)
    if nonzero_indices.shape[0] == 0:
        return None, None, None, None, None, None
    ## get boundary
    min_d, max_d = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
    min_h, max_h = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
    min_w, max_w = nonzero_indices[:, 2].min(), nonzero_indices[:, 2].max()
    ## padding
    crop_d, crop_h, crop_w = max_d - min_d + 1, max_h - min_h + 1, max_w - min_w + 1,
    window_d, window_h, window_w = spatial_size
    padding_d, padding_h, padding_w = max(0, window_d-crop_d), max(0, window_h-crop_h), max(0, window_w-crop_w)
    global_d, global_h, global_w = logits_global_single.shape
    min_d = max(0, min_d - int(padding_d)//2)
    min_h = max(0, min_h - int(padding_h)//2)
    min_w = max(0, min_w - int(padding_w)//2)
    max_d = min(global_d, max_d + int(padding_d)//2)
    max_h = min(global_h, max_h + int(padding_h)//2)
    max_w = min(global_w, max_w + int(padding_w)//2)
    return min_d, min_h, min_w, max_d, max_h, max_w

def build_binary_cube(bbox, binary_cube_shape):
    min_coord = bbox[0][:3].int().tolist()
    max_coord = bbox[0][3:].int().tolist()
    binary_cube = torch.zeros(binary_cube_shape)
    binary_cube[min_coord[0]:max_coord[0]+1, min_coord[1]:max_coord[1]+1, min_coord[2]:max_coord[2]+1] = 1
    return binary_cube

def build_binary_points(points, labels, shape):
    binary_points = torch.zeros(shape, dtype=torch.int16)
    binary_points[points[labels == 1, 0].long(), points[labels == 1, 1].long(), points[labels == 1, 2].long()] = 1
    return binary_points

def sliding_window_inference(
    inputs: torch.Tensor,
    prompt_reflection: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    print('sliding window inference for ROI')
    text = kwargs['text']
    use_box = kwargs['use_box']
    use_point = kwargs['use_point']
    assert not (use_box and use_point)
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
    #############
    if use_point or use_box:
        binary_prompt_map, global_preds = prompt_reflection
        global_preds = F.pad(global_preds, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
    #############
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True  # whether the predictor's output is a tensor (instead of dict/tuple)

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        #############
        
        boxes = None
        points = None
        if use_point:
            window_binary_prompt_map = torch.cat([binary_prompt_map[win_slice] for win_slice in unravel_slice]).to(sw_device)
            point, point_label = select_points(window_binary_prompt_map.squeeze())
            points = (point.unsqueeze(0).float().cuda(), point_label.unsqueeze(0).float().cuda())  
            pseudo_label = torch.cat([global_preds[win_slice] for win_slice in unravel_slice]).to(sw_device)
            boxes = generate_box(pseudo_label.squeeze()).unsqueeze(0).float().cuda()
        if use_box:
            if num_win == 1:
                window_binary_prompt_map = torch.cat([binary_prompt_map[win_slice] for win_slice in unravel_slice]).to(sw_device)
                boxes = generate_box(window_binary_prompt_map.squeeze()).unsqueeze(0).float().cuda()
            else:
                pseudo_label = torch.cat([global_preds[win_slice] for win_slice in unravel_slice]).to(sw_device)
                boxes = generate_box(pseudo_label.squeeze()).unsqueeze(0).float().cuda()
        seg_prob_out = predictor(window_data, text, boxes, points)  # batched patch segmentation
        #############
        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    return final_output[0] if is_tensor_output else final_output  # type: ignore


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def generate_box(pred_pre, bbox_shift=None):
    meaning_post_label = pred_pre # [h, w, d]
    ones_idx = (meaning_post_label > 0).nonzero(as_tuple=True)
    if all(tensor.nelement() == 0 for tensor in ones_idx):
        bboxes = torch.tensor([-1,-1,-1,-1,-1,-1])
        # print(bboxes, bboxes.shape)
        return bboxes
    min_coords = [dim.min() for dim in ones_idx]    # [x_min, y_min, z_min]
    max_coords = [dim.max() for dim in ones_idx]    # [x_max, y_max, z_max]


    if bbox_shift is None:
        corner_min = []
        corner_max = []
        shape = meaning_post_label.shape
        for coor in min_coords:
            coor_ = max(0, coor)
            corner_min.append(coor_)
        for idx, coor in enumerate(max_coords):
            coor_ = min(shape[idx], coor)
            corner_max.append(coor_)
        corner_min = torch.tensor(corner_min)
        corner_max = torch.tensor(corner_max)
        return torch.cat((corner_min, corner_max), dim=0)
    else:
        # add perturbation to bounding box coordinates
        corner_min = []
        corner_max = []
        shape = meaning_post_label.shape
        for coor in min_coords:
            coor_ = max(0, coor + random.randint(-bbox_shift, bbox_shift))
            corner_min.append(coor_)
        for idx, coor in enumerate(max_coords):
            coor_ = min(shape[idx], coor + random.randint(-bbox_shift, bbox_shift))
            corner_max.append(coor_)
        corner_min = torch.tensor(corner_min)
        corner_max = torch.tensor(corner_max)
        return torch.cat((corner_min, corner_max), dim=0)


def select_points(preds, num_positive_extra=4, num_negative_extra=0, fix_extra_point_num=None):
    spacial_dim = 3
    points = torch.zeros((0, 3))
    labels = torch.zeros((0))
    pos_thred = 0.9
    neg_thred = 0.1
    
    # get pos/net indices
    positive_indices = torch.nonzero(preds > pos_thred, as_tuple=True) # ([pos x], [pos y], [pos z])
    negative_indices = torch.nonzero(preds < neg_thred, as_tuple=True)

    ones_idx = (preds > pos_thred).nonzero(as_tuple=True)
    if all(tmp.nelement() == 0 for tmp in ones_idx):
        # all neg
        num_positive_extra = 0
        selected_positive_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))
    else:
        # random select a pos point
        random_idx = torch.randint(len(positive_indices[0]), (1,))
        selected_positive_point = torch.tensor([positive_indices[i][random_idx] for i in range(spacial_dim)]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.ones((1))))

    if num_positive_extra > 0:
        pos_idx_list = torch.randperm(len(positive_indices[0]))[:num_positive_extra]
        extra_positive_points = []
        for pos_idx in pos_idx_list:
            extra_positive_points.append([positive_indices[i][pos_idx] for i in range(spacial_dim)])
        extra_positive_points = torch.tensor(extra_positive_points).reshape(-1, 3)
        points = torch.cat((points, extra_positive_points), dim=0)
        labels = torch.cat((labels, torch.ones((extra_positive_points.shape[0]))))

    if num_negative_extra > 0:
        neg_idx_list = torch.randperm(len(negative_indices[0]))[:num_negative_extra]
        extra_negative_points = []
        for neg_idx in neg_idx_list:
            extra_negative_points.append([negative_indices[i][neg_idx] for i in range(spacial_dim)])
        extra_negative_points = torch.tensor(extra_negative_points).reshape(-1, 3)
        points = torch.cat((points, extra_negative_points), dim=0)
        labels = torch.cat((labels, torch.zeros((extra_negative_points.shape[0]))))
        # print('extra_negative_points ', extra_negative_points, extra_negative_points.shape)
        # print('==> points ', points.shape, labels)
    
    if fix_extra_point_num is None:
        left_point_num = num_positive_extra + num_negative_extra + 1 - labels.shape[0]
    else:
        left_point_num = fix_extra_point_num  + 1 - labels.shape[0]

    for _ in range(left_point_num):
        ignore_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, ignore_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))

    return (points, labels)
