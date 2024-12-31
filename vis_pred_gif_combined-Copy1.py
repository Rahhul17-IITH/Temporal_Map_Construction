import argparse
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

import tqdm
import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize

import os
os.environ['TORCH_HOME'] = os.path.expanduser('~/torch_cache')


def create_overlapped_visualization(model, val_loader, angle_class):
    model.eval()
    car_img = Image.open('icon/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
                    yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                               post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                               lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            
            semantic_vis = semantic.softmax(1).cpu().numpy()
            semantic_vis[semantic_gt < 0.1] = np.nan

            for si in range(semantic.shape[0]):
                plt.figure(figsize=(4, 2))
                
                # Plot segmentation first
                plt.imshow(semantic_vis[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.3)
                plt.imshow(semantic_vis[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.3)
                plt.imshow(semantic_vis[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.3)
                
                # Plot vector predictions on top
                coords, types, _ = vectorize(semantic[si], embedding[si], direction[si], angle_class)
                colors = ['red', 'blue', 'green']
                for coord, type_idx in zip(coords, types):
                    # Convert type_idx to integer
                    type_idx_int = int(type_idx) if isinstance(type_idx, (np.floating, float)) else type_idx
                    # Ensure type_idx is within bounds
                    type_idx_int = min(max(0, type_idx_int), len(colors)-1)
                    plt.plot(coord[:, 0], coord[:, 1], color=colors[type_idx_int], linewidth=3)

                # Add car image
                plt.imshow(car_img, extent=[semantic.shape[3]//2-15, semantic.shape[3]//2+15,
                                          semantic.shape[2]//2-12, semantic.shape[2]//2+12])
                
                plt.xlim(0, semantic.shape[3])
                plt.ylim(semantic.shape[2], 0)
                plt.axis('off')

                img_name = f'overlap_eval{batchi:06}_{si:03}.jpg'
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()

    # Create GIF
    create_gif('overlap_eval*.jpg', 'hdmapnet_overlapped.gif')


def create_combined_visualization(model, val_loader, angle_class):
    model.eval()
    car_img = Image.open('icon/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
                    yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                               post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                               lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            
            semantic_vis = semantic.softmax(1).cpu().numpy()
            semantic_vis[semantic_gt < 0.1] = np.nan

            for si in range(semantic.shape[0]):
                # Create a figure with two subplots side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
                
                # Left subplot - Segmentation
                ax1.imshow(semantic_vis[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                ax1.imshow(semantic_vis[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                ax1.imshow(semantic_vis[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
                ax1.set_xlim(0, 400)
                ax1.set_ylim(200, 0)
                ax1.axis('off')
                ax1.set_title('Segmentation')
                
                # Right subplot - Vector Prediction
                coords, _, _ = vectorize(semantic[si], embedding[si], direction[si], angle_class)
                for coord in coords:
                    ax2.plot(coord[:, 0], coord[:, 1], linewidth=5)
                ax2.imshow(car_img, extent=[semantic.shape[3]//2-15, semantic.shape[3]//2+15,
                                          semantic.shape[2]//2-12, semantic.shape[2]//2+12])
                ax2.set_xlim(0, semantic.shape[3])
                ax2.set_ylim(semantic.shape[2], 0)
                ax2.axis('off')
                ax2.set_title('Prediction')

                plt.tight_layout()
                
                # Save combined image
                img_name = f'combined_eval{batchi:06}_{si:03}.jpg'
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()

    # Create single combined GIF
    create_gif('combined_eval*.jpg', 'hdmapnet_combined.gif')



def create_gif(image_pattern, output_gif):
    # Get list of images in sorted order
    image_files = sorted(glob.glob(image_pattern))
    if not image_files:
        return
    
    # Open all images
    images = [Image.open(f) for f in image_files]
    
    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=500,  # Duration for each frame in milliseconds
        loop=0
    )
    
    # Clean up individual images
    for img_file in image_files:
        os.remove(img_file)

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis_segmentation(model, val_loader):
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            semantic = semantic.softmax(1).cpu().numpy()
            semantic[semantic_gt < 0.1] = np.nan

            for si in range(semantic.shape[0]):
                plt.figure(figsize=(4, 2))
                plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                plt.close()
         # After the loop ends, create GIF
    create_gif('eval*.jpg', 'hdmapnet_segmentation.gif')




def vis_vector(model, val_loader, angle_class):
    model.eval()
    car_img = Image.open('icon/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt) in enumerate(val_loader):

            segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                       post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                       lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            for si in range(segmentation.shape[0]):
                coords, _, _ = vectorize(segmentation[si], embedding[si], direction[si], angle_class)

                for coord in coords:
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)

                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))
                plt.imshow(car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])

                img_name = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()
     # After the loop ends, create GIF
    create_gif('eval*.jpg', 'hdmapnet_prediction.gif')



def main(args):
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
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    vis_vector(model, val_loader, args.angle_class)
    vis_segmentation(model, val_loader)
    create_combined_visualization(model, val_loader, args.angle_class)
    create_overlapped_visualization(model, val_loader, args.angle_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
