import argparse
import os
import torch
import torch.nn.functional as F
import json
from segment_anything_volumetric import sam_model_registry
from network.model import SegVol
from data_process.demo_data_process import process_ct_gt
import monai.transforms as transforms
from utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor
from utils.visualize import draw_result

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument("--resume", type = str, default = '')
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    ### demo
    parser.add_argument('--demo_config', type=str, required=True)
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')
    args = parser.parse_args()
    return args

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match\n" + str(preds.shape) + str(labels.shape)
    predict = preds.view(1, -1)
    target = labels.view(1, -1)
    if target.shape[1] < 1e8:
        predict = predict.cuda()
        target = target.cuda()
    predict = torch.sigmoid(predict)
    predict = torch.where(predict > 0.5, 1., 0.)
    
    tp = torch.sum(torch.mul(predict, target))
    den = torch.sum(predict) + torch.sum(target) + 1
    dice = 2 * tp / den

    if target.shape[1] < 1e8:
        predict = predict.cpu()
        target = target.cpu()
    return dice

def zoom_in_zoom_out(args, segvol_model, image, image_resize, gt3D, gt3D_resize, categories=None):
    logits_labels_record = {}
    image_single_resize = image_resize
    image_single = image[0,0]
    ori_shape = image_single.shape
    for item_idx in range(len(categories)):
        # get label to generate prompts
        label_single = gt3D[0][item_idx]
        label_single_resize = gt3D_resize[0][item_idx]
        # skip meaningless categories
        if torch.sum(label_single) == 0:
            print('No object, skip')
            continue
        # generate prompts
        text_single = categories[item_idx] if args.use_text_prompt else None
        if categories is not None: print(f'inference |{categories[item_idx]}| target...')
        points_single = None
        box_single = None
        if args.use_point_prompt:
            point, point_label = select_points(label_single_resize, num_positive_extra=3, num_negative_extra=3)
            points_single = (point.unsqueeze(0).float().cuda(), point_label.unsqueeze(0).float().cuda()) 
            binary_points_resize = build_binary_points(point, point_label, label_single_resize.shape)
        if args.use_box_prompt:
            box_single = generate_box(label_single_resize).unsqueeze(0).float().cuda()
            binary_cube_resize = build_binary_cube(box_single, binary_cube_shape=label_single_resize.shape)
        
        ####################
        # zoom-out inference:
        print('--- zoom out inference ---')
        print(f'use text-prompt [{text_single!=None}], use box-prompt [{box_single!=None}], use point-prompt [{points_single!=None}]')
        with torch.no_grad():
            logits_global_single = segvol_model(image_single_resize.cuda(),
                                                text=text_single, 
                                                boxes=box_single, 
                                                points=points_single)
        
        # resize back global logits
        logits_global_single = F.interpolate(
                logits_global_single.cpu(),
                size=ori_shape, mode='nearest')[0][0]
        
        # build prompt reflection for zoom-in
        if args.use_point_prompt:
            binary_points = F.interpolate(
                binary_points_resize.unsqueeze(0).unsqueeze(0).float(),
                size=ori_shape, mode='nearest')[0][0]
        if args.use_box_prompt:
            binary_cube = F.interpolate(
                binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
                size=ori_shape, mode='nearest')[0][0]
        zoom_out_dice = dice_score(logits_global_single.squeeze(), label_single.squeeze())
        logits_labels_record[categories[item_idx]] = (
            zoom_out_dice,
            image_single, 
            points_single,
            box_single,
            logits_global_single, 
            label_single)
        print(f'zoom out inference done with zoom_out_dice: {zoom_out_dice:.4f}')
        if not args.use_zoom_in:
            continue

        ####################
        # zoom-in inference:
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(args.spatial_size, logits_global_single)
        if min_d is None:
            print('Fail to detect foreground!')
            continue

        # Crop roi
        image_single_cropped = image_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
        global_preds = (torch.sigmoid(logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>0.5).long()
        
        assert not (args.use_box_prompt and args.use_point_prompt)
        prompt_reflection = None
        if args.use_box_prompt:
            binary_cube_cropped = binary_cube[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )
        if args.use_point_prompt:
            binary_points_cropped = binary_points[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                binary_points_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )
    
        ## inference
        with torch.no_grad():
            logits_single_cropped = sliding_window_inference(
                    image_single_cropped.cuda(), prompt_reflection,
                    args.spatial_size, 1, segvol_model, args.infer_overlap,
                    text=text_single,
                    use_box=args.use_box_prompt,
                    use_point=args.use_point_prompt,
                )
            logits_single_cropped = logits_single_cropped.cpu().squeeze()
        logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_single_cropped
        zoom_in_dice = dice_score(logits_global_single.squeeze(), label_single.squeeze())
        logits_labels_record[categories[item_idx]] = (
            zoom_in_dice,
            image_single, 
            points_single,
            box_single,
            logits_global_single, 
            label_single)
        print(f'===> zoom out dice {zoom_out_dice:.4f} -> zoom-out-zoom-in dice {zoom_in_dice:.4f} <===')
    return logits_labels_record

def inference_single_ct(args, segvol_model, data_item, categories):
    segvol_model.eval()
    image, gt3D = data_item["image"].float(), data_item["label"]
    image_zoom_out, gt3D__zoom_out = data_item["zoom_out_image"].float(), data_item['zoom_out_label']

    logits_labels_record = zoom_in_zoom_out(
        args, segvol_model, 
        image.unsqueeze(0), image_zoom_out.unsqueeze(0), 
        gt3D.unsqueeze(0), gt3D__zoom_out.unsqueeze(0),
        categories=categories)
    
    # visualize
    if args.visualize:
        for target, values in logits_labels_record.items():
            dice_score, image, point_prompt, box_prompt, logits, labels = values
            print(f'{target} result with Dice score {dice_score:.4f} visualizing')
            draw_result(target + f"-Dice {dice_score:.4f}", image, box_prompt, point_prompt, logits, labels, args.spatial_size, args.work_dir)

def main(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    # build model
    sam_model = sam_model_registry['vit'](args=args)
    segvol_model = SegVol(
                        image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        ).cuda()
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])

    # load param
    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        segvol_model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # load demo config
    with open(args.demo_config, 'r') as file:
        config_dict = json.load(file)
    ct_path, gt_path, categories = config_dict['demo_case']['ct_path'], config_dict['demo_case']['gt_path'], config_dict['categories']

    # preprocess for data
    data_item = process_ct_gt(ct_path, gt_path, categories, args.spatial_size)

    # seg config for prompt & zoom-in-zoom-out
    args.use_zoom_in = True
    args.use_text_prompt = True
    args.use_box_prompt = True
    args.use_point_prompt = False
    args.visualize = True

    inference_single_ct(args, segvol_model, data_item, categories)

if __name__ == "__main__":
    args = set_parse()
    main(args)
