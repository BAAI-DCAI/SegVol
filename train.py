import os
import torch
import argparse
from datetime import datetime
from network.model import SegVol
from segment_anything_volumetric import sam_model_registry
import torch.multiprocessing as mp
import shutil
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import BCELoss, BinaryDiceLoss
from data_utils import get_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm

def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--pretrain", type = str, default='')
    parser.add_argument("--resume", type = str, default='')
    parser.add_argument("--data_dir", type = str, default='')
    parser.add_argument("--dataset_codes", type = list, default=['0010', '0011'])
    # config
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument('-num_workers', type=int, default=8)
    # dist
    parser.add_argument('--dist', dest='dist', type=bool, default=True,
                        help='distributed training or not')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--init_method', type = str, default = "env://")
    parser.add_argument('--bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    # key params
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    parser.add_argument('-warmup_epoch', type=int, default=10)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument("--use_pseudo_label", default=True, type=bool)
    args = parser.parse_args()
    return args

def train_epoch(args, segvol_model, train_dataloader, optimizer, scheduler, epoch, rank, gpu, iter_num):
    epoch_loss = 0
    epoch_sl_loss = 0
    epoch_ssl_loss = 0

    epoch_iterator = tqdm(
        train_dataloader, desc = f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True
    )
    if args.dist:
        train_dataloader.sampler.set_epoch(epoch)
        torch.distributed.barrier()
    
    for batch in epoch_iterator:
        image, gt3D = batch["image"].cuda(), batch["post_label"].cuda()
        pseudo_seg_cleaned = batch['pseudo_seg_cleaned'].cuda()
        organ_name_list = batch['organ_name_list']

        loss_step_avg = 0
        sl_loss_step_avg = 0
        ssl_loss_step_avg = 0
        for cls_idx in range(len(organ_name_list)):
            optimizer.zero_grad()
            organs_cls = organ_name_list[cls_idx]
            labels_cls = gt3D[:, cls_idx]

            if torch.sum(labels_cls) == 0:
                print(f'[RANK {rank}: GPU {gpu}] ITER-{iter_num} --- No object, skip iter')
                continue

            sl_loss, ssl_loss = segvol_model(image, organs=None, boxes=None, points=None,
                                            train_organs=organs_cls,
                                            train_labels=labels_cls,
                                            pseudo_seg_cleaned=pseudo_seg_cleaned)
            if args.use_pseudo_label:
                loss = sl_loss + 0.1 * ssl_loss
                ssl_loss_step_avg += ssl_loss.item()
                sl_loss_step_avg += sl_loss.item()
            loss_step_avg += loss.item()
            
            loss.backward()
            optimizer.step()
            print(f'[RANK {rank}: GPU {gpu}] ITER-{iter_num} --- loss {loss.item()}, sl_loss, {sl_loss.item()}, ssl_loss {ssl_loss.item()}')
            iter_num += 1

        loss_step_avg /= len(organ_name_list)
        sl_loss_step_avg /= len(organ_name_list)
        ssl_loss_step_avg /= len(organ_name_list)
        print(f'[RANK {rank}: GPU {gpu}] AVG loss {loss_step_avg}, sl_loss, {sl_loss_step_avg}, ssl_loss {ssl_loss_step_avg}')
        if rank == 0:
            args.writer.add_scalar('train_iter/loss', loss_step_avg, iter_num)
            args.writer.add_scalar('train_iter/sl_loss', sl_loss_step_avg, iter_num)
            args.writer.add_scalar('train_iter/ssl_loss', ssl_loss_step_avg, iter_num)

        epoch_loss += loss_step_avg
        epoch_sl_loss += sl_loss_step_avg
        if args.use_pseudo_label:
            epoch_ssl_loss += ssl_loss_step_avg
    scheduler.step() 
    epoch_loss /= len(train_dataloader) + 1e-12
    epoch_ssl_loss /= len(train_dataloader) + 1e-12
    epoch_sl_loss /= len(train_dataloader) + 1e-12
    print(f'{args.model_save_path} ==> [RANK {rank}: GPU {gpu}] ', 'epoch_loss: {}, ssl_loss: {}'.format(epoch_loss, epoch_ssl_loss))
    if rank == 0:
        args.writer.add_scalar('train/loss', epoch_loss, epoch)
        args.writer.add_scalar('train/sl_loss', epoch_sl_loss, epoch)
        args.writer.add_scalar('train/ssl_loss', epoch_ssl_loss, epoch)
        args.writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, iter_num

def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node #args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(args.model_save_path, args.run_id + '_' + os.path.basename(__file__)))
    torch.cuda.set_device(gpu)
    
    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = args.init_method,
        rank = rank,
        world_size = world_size,
    )
    print('init_process_group finished')

    sam_model = sam_model_registry['vit'](args=args, checkpoint=None)   # checkpoint for pretrained vit
    segvol_model = SegVol(
                        image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        ).cuda()
    
    segvol_model = torch.nn.parallel.DistributedDataParallel(
        segvol_model,
        device_ids = [gpu],
        output_device = gpu,
        gradient_as_bucket_view = True,
        find_unused_parameters = True,
        bucket_cap_mb = args.bucket_cap_mb
    )

    optimizer = torch.optim.AdamW(
        segvol_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs)

    #%% train
    num_epochs = args.num_epochs
    iter_num = 0

    train_dataloader = get_loader(args)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location = loc)
            segvol_model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        torch.distributed.barrier()

    if rank == 0:
        args.writer = SummaryWriter(log_dir='./tb_log/' + args.run_id)
        print('Writing Tensorboard logs to ', './tb_log/' + args.run_id)

    for epoch in range(start_epoch, num_epochs):
        with segvol_model.join():
            epoch_loss, iter_num = train_epoch(args, segvol_model, train_dataloader, optimizer, scheduler, epoch, rank, gpu, iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host and (epoch+1) % 10 == 0:
            checkpoint = {
                'model': segvol_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.model_save_path, f'medsam_model_e{epoch+1}.pth'))
        torch.distributed.barrier()

def main():
    # set seeds
    torch.manual_seed(2023)
    torch.cuda.empty_cache()
    args = set_parse()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12222'
    if args.use_pseudo_label:
        print('----- use pseudo_label -----')
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    print(f"=====> project save at {args.model_save_path}")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()
