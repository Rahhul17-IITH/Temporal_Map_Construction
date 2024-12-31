#Implementation for ONNX
import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse
import torch
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from loss import SimpleLoss, DiscriminativeLoss
from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou
from torch.nn import BCEWithLogitsLoss
from torch import nn
from torch.optim.lr_scheduler import LinearLR


# Set cache directories to user's home directory
os.environ['TORCH_HOME'] = os.path.expanduser('~/torch_cache')


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)
    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)

class ONNXSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def validate_inputs(batch):
    for k, v in batch.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"Found NaN or Inf in {k}")


def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"), filemode='w', format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)

    if args.finetune:
        model.load_state_dict(torch.load(args.modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    warmup_scheduler = LinearLR(opt, start_factor=0.1,total_iters=10)
    writer = SummaryWriter(logdir=args.logdir)
    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = BCEWithLogitsLoss(reduction='none')
    scaler = GradScaler()

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(train_loader):
            t0 = time()
            
            with autocast():
                semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(), post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(), lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
                
                semantic_gt = semantic_gt.cuda().float()
                instance_gt = instance_gt.cuda()
                
                seg_loss = loss_fn(semantic, semantic_gt)
                
                if args.instance_seg:
                    var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
                else:
                    var_loss = dist_loss = reg_loss = 0
                
                if args.direction_pred:
                    direction_gt = direction_gt.cuda()
                    lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                    direction_loss = direction_loss_fn(direction, direction_gt)
                    direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                    angle_diff = calc_angle_diff(torch.softmax(direction, 1), direction_gt, args.angle_class)
                else:
                    direction_loss = 0
                    angle_diff = 0
                
                final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction
                final_loss = final_loss / args.grad_accum_steps

            scaler.scale(final_loss).backward()
            
            if (batchi + 1) % args.grad_accum_steps == 0 or batchi == last_idx:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            counter += 1
            t1 = time()
            
            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}] "
                            f"Time: {t1-t0:>7.4f} "
                            f"Loss: {final_loss.item():>7.4f} "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)

        # iou = eval_iou(model, val_loader)
        # logger.info(f"EVAL[{epoch:>2d}]: "
        #             f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        # write_log(writer, iou, 'eval', counter)
        try:
            iou = eval_iou(model, val_loader)
            if iou is not None:
                logger.info(f"EVAL[{epoch:>2d}]: IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
                write_log(writer, iou, 'eval', counter)
            else:
                logger.warning(f"EVAL[{epoch:>2d}]: No valid IOU computed")
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            
        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")
        model.train()
        sched.step()
        # After training loop ends, export to ONNX
        logger.info("Exporting model to ONNX format...")
        onnx_path = os.path.join(args.logdir, "hdmapnet_final.onnx")
        export_to_onnx(model, data_conf, onnx_path)


        
def export_to_onnx(model, data_conf, save_path='hdmapnet.onnx'):
    model.eval()
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"), filemode='w', format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # Replace Swish implementations before export
    def replace_swish(module):
        if hasattr(module, '_swish'):
            module._swish = ONNXSwish()
        if hasattr(module, 'trunk'):
            if hasattr(module.trunk, '_swish'):
                module.trunk._swish = ONNXSwish()
            for block in module.trunk._blocks:
                if hasattr(block, '_swish'):
                    block._swish = ONNXSwish()
        for child in module.children():
            replace_swish(child)
    
    # Replace Swish implementations
    replace_swish(model)
    
    # Create sample inputs matching the model's expected dimensions
    batch_size = 1
    sample_inputs = {
        'imgs': torch.randn(batch_size, 6, 3, data_conf['image_size'][0], data_conf['image_size'][1]),
        'trans': torch.randn(batch_size, 6, 3),
        'rots': torch.randn(batch_size, 6, 3, 3),
        'intrins': torch.randn(batch_size, 6, 3, 3),
        'post_trans': torch.randn(batch_size, 6, 3),
        'post_rots': torch.randn(batch_size, 6, 3, 3),
        'lidar_data': torch.randn(batch_size, 1, 64, 64, 3),
        'lidar_mask': torch.ones(batch_size, 1, 64, 64),
        'car_trans': torch.randn(batch_size, 3),
        'yaw_pitch_roll': torch.randn(batch_size, 3)
    }
    
    # Move inputs to CUDA if available
    device = next(model.parameters()).device
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}

    try:
        torch.onnx.export(
            model,
            (sample_inputs['imgs'], sample_inputs['trans'], sample_inputs['rots'],
             sample_inputs['intrins'], sample_inputs['post_trans'], sample_inputs['post_rots'],
             sample_inputs['lidar_data'], sample_inputs['lidar_mask'],
             sample_inputs['car_trans'], sample_inputs['yaw_pitch_roll']),
            save_path,
            input_names=['imgs', 'trans', 'rots', 'intrins', 'post_trans', 'post_rots',
                        'lidar_data', 'lidar_mask', 'car_trans', 'yaw_pitch_roll'],
            output_names=['semantic', 'embedding', 'direction'],
            dynamic_axes={
                'imgs': {0: 'batch_size'},
                'trans': {0: 'batch_size'},
                'rots': {0: 'batch_size'},
                'intrins': {0: 'batch_size'},
                'post_trans': {0: 'batch_size'},
                'post_rots': {0: 'batch_size'},
                'lidar_data': {0: 'batch_size'},
                'lidar_mask': {0: 'batch_size'},
                'car_trans': {0: 'batch_size'},
                'yaw_pitch_roll': {0: 'batch_size'},
                'semantic': {0: 'batch_size'},
                'embedding': {0: 'batch_size'},
                'direction': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True
        )
        if logger:
            logger.info(f"Successfully exported model to {save_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to export model to ONNX: {str(e)}")
        raise


def validate_onnx(onnx_path, data_loader):
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    
    # Run inference on a batch
    for batch in data_loader:
        imgs, trans, rots, intrins, post_trans, post_rots, \
        lidar_data, lidar_mask, car_trans, yaw_pitch_roll, _, _, _ = batch
        
        # Prepare input dict
        onnx_inputs = {
            'imgs': imgs.numpy(),
            'trans': trans.numpy(),
            'rots': rots.numpy(),
            'intrins': intrins.numpy(),
            'post_trans': post_trans.numpy(),
            'post_rots': post_rots.numpy(),
            'lidar_data': lidar_data.numpy(),
            'lidar_mask': lidar_mask.numpy(),
            'car_trans': car_trans.numpy(),
            'yaw_pitch_roll': yaw_pitch_roll.numpy()
        }
        
        # Run inference
        outputs = session.run(None, onnx_inputs)
        return outputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    parser.add_argument("--logdir", type=str, default='./runs')
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--model", type=str, default='HDMapNet_cam')
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format after training')
    parser.add_argument('--onnx_path', type=str, default=None, help='Path to save ONNX model')
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    args = parser.parse_args()
    train(args)