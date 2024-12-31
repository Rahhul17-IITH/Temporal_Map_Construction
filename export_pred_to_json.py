import argparse
import mmcv
import tqdm
import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
import onnxruntime as ort
import os
os.environ['TORCH_HOME'] = os.path.expanduser('~/torch_cache')

import numpy as np


def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]]
    return dx, bx, nx

def export_to_json(model, val_loader, angle_class, args):
    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {}
    }

    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt) in enumerate(tqdm.tqdm(val_loader)):
            if isinstance(model, ort.InferenceSession):
                input_dict = {
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
                outputs = model.run(None, input_dict)
                segmentation, embedding, direction = [torch.from_numpy(out).cuda() for out in outputs]



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
    model = get_model(args.model, data_conf, True, args.embedding_dim, True, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    export_to_json(model, val_loader, args.angle_class, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument('--onnx_path', type=str, default=None, help='Path to ONNX model')


    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument("--embedding_dim", type=int, default=16)

    # direction config
    parser.add_argument('--angle_class', type=int, default=36)

    # output
    parser.add_argument("--output", type=str, default='output.json')

    args = parser.parse_args()
    main(args)
