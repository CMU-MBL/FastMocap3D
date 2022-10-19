import os, sys
import os.path as osp
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from lib.models.builder import get_network, load_checkpoint
from lib.models.smpl import build_body_model
from lib.data.amass import setup_amass_dloader
from lib.utils.criterion import Criterion
from lib.utils.checkpoint import Logger
from lib.data import data_utils as d_utils
from configs.train_options import TrainOptions
from configs import constants as _C

from pdb import set_trace as st


test_smpl = None
def validate_curr_epoch(args, net, test_dloader, criterion, epoch):
    global test_smpl
    if test_smpl is None: test_smpl = build_body_model(args, batch_size=args.seq_length)
    net.eval()

    (mpjpes, mpjpes_a, mpjpes_pa, mpjaes) = [defaultdict(list) for _ in range(4)]
    with torch.no_grad():
        for iter_, batch in enumerate(test_dloader):
            inputs, groundtruths = d_utils.prepare_batch(args, batch)
            pred_smpl, _ = net(inputs)

            output = test_smpl.get_output(pred_smpl, groundtruths)
            pred_keypoints = output.joints.view(
                1, args.seq_length, -1, 3)

            mpjpe = criterion._get_mpjpe(groundtruths['keypoints'], pred_keypoints,
                                         align=False, return_batch=True)
            mpjpe_a = criterion._get_mpjpe(groundtruths['keypoints'], pred_keypoints,
                                           align=True, return_batch=True)
            mpjpe_pa = criterion._get_pa_mpjpe(groundtruths['keypoints'], pred_keypoints,
                                               return_batch=True)
            mpjae = criterion._get_mpjae(groundtruths['pose'], pred_smpl[0])

            for (key, vals, val) in zip(
                        ['overall'] * 4 + [batch['vid_name'][0]] * 4,
                        (mpjpes, mpjpes_a, mpjpes_pa, mpjaes) * 2,
                        (mpjpe[0], mpjpe_a[0], mpjpe_pa[0], mpjae[0]) * 2):
                vals[key].append(val.item())
            if 'acting3' in batch['vid_name'][0]:
                mpjaes['overall'].pop()
                mpjaes[batch['vid_name'][0]][-1] = 0

    for k in mpjpes.keys():
        print(f'{k} | MPJPE: {np.array(mpjpes[k]).mean():.2f} (mm)' + \
              f'   MPJPE (Aligned): {np.array(mpjpes_a[k]).mean():.2f} (mm)' + \
              f'   PA-MPJPE: {np.array(mpjpes_pa[k]).mean():.2f} (mm)' + \
              f'   MPJAE: {np.array(mpjaes[k]).mean():.2f} (deg)')

    walking_mpjaes = []
    for k in mpjaes.keys():
        if 'walking' in k: walking_mpjaes.append(np.array(mpjaes[k]))
    walking_mpjaes = np.concatenate(walking_mpjaes, 0).mean()

    return {'epoch': epoch, 'mpjpe': np.array(mpjpes['overall']).mean(),
            'mpjpe_a': np.array(mpjpes_a['overall']).mean(),
            'mpjpe_pa': np.array(mpjpes_pa['overall']).mean(),
            'mpjae': walking_mpjaes}


train_smpl = None
def train_one_epoch(args, net, train_dloader, optimizer, criterion, logger, epoch):
    global train_smpl
    if train_smpl is None: train_smpl = build_body_model(args)

    net.train()
    for _iter, batch in enumerate(train_dloader):
        if batch['pose'].shape[0] != args.batch_size: continue

        inputs, groundtruths = d_utils.prepare_batch(args, batch)
        pred_smpl, outs = net(inputs)

        output = train_smpl.get_output(pred_smpl, groundtruths)
        pred_keypoints = output.joints.view(args.batch_size, args.seq_length, -1, 3)

        total_loss, loss_dict = criterion(pred_smpl, pred_keypoints, groundtruths)
        if args.model_type != 'imu':
            loss_dict['Input Keypoints Qty'] = criterion._get_mpjpe(
                inputs['keypoints'], groundtruths['keypoints'])

        if args.model_type == 'fusion' and epoch < args.supervise_unimodal_until_n_epoch:
            loss_v, loss_dict_v = criterion((outs[0], None, None), None, groundtruths)
            loss_i, loss_dict_i = criterion((outs[1], None, None), None, groundtruths)
            total_loss += (loss_v + loss_i)

            for key, val in loss_dict_v.items():
                loss_dict[f'{key} (Video)'] = val
            for key, val in loss_dict_i.items():
                loss_dict[f'{key} (IMU)'] = val

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        logger(loss_dict, _iter, epoch)



def main(start_epoch):
    args = TrainOptions().parse_args()
    net = get_network(args)

    train_dloader, test_dloader = setup_amass_dloader(args)
    criterion = Criterion(args)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=(args.betas, 0.999))

    if osp.exists(args.init_weight):
        net, optimizer, start_epoch = load_checkpoint(args, net, optimizer)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                  factor=0.1, verbose=True)

    logger = Logger(model_dir=osp.join(args.logdir, args.name),
                    write_freq=args.write_freq,
                    checkpoint_freq=args.checkpoint_freq,
                    total_iteration=len(train_dloader))
    logger._save_configs(args)

    for epoch in range(start_epoch, args.epoch + 1):
        train_one_epoch(args, net, train_dloader, optimizer, criterion, logger, epoch)
        val_results = validate_curr_epoch(args, net, test_dloader, criterion, epoch)
        logger._save_checkpoint(net, {'optimizer': optimizer},
                                eval_dict=val_results, epoch=epoch)
        train_dloader.dataset.prepare_video_batch()


if __name__ == '__main__':
    main(start_epoch=1)
