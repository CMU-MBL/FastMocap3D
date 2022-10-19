from collections import defaultdict

import torch
import numpy as np

from lib.models.smpl import build_body_model
from lib.utils.conversion import angle_axis_to_rotation_matrix
from lib.utils.smpl_utils import (smpl_pose_to_acceleration,
                                  smpl_pose_to_imu_orientation,
                                  smpl_pose_to_angular_velocity,
                                  smpl_pose_to_global_pose)
from configs import constants as _C


def make_collate_fn():
    def collate_fn(items):
        items = list(filter(lambda x: x is not None , items))
        batch = dict()
        batch['vid_name'] = [item['vid_name'] for item in items]
        for key in items[0].keys():
            try: batch[key] = torch.stack([item[key] for item in items])
            except: pass
        return batch

    return collate_fn


def prepare_batch(args, batch):
    inputs = defaultdict()
    groundtruths = defaultdict()

    gt_pose = batch['pose'].to(device=args.device)[:, :, :-2]
    groundtruths['pose'] = gt_pose
    groundtruths['betas'] = batch['betas'].to(device=args.device)
    groundtruths['keypoints'] = batch['gt_keypoints'].to(device=args.device)
    groundtruths['transl'] = batch['transl'].to(device=args.device)

    try:
        input_keypoints = batch['input_keypoints'].to(device=args.device)
        input_keypoints, conf = input_keypoints[..., :-1], input_keypoints[..., -1:]
        inputs['conf'] = conf
        inputs['keypoints'] = input_keypoints
    except: pass

    try:
        gyr = batch['gyr'].to(device=args.device)
        acc = batch['acc'].to(device=args.device)
        inputs['imu'] = torch.cat((gyr, acc), dim=-1)
    except: pass

    return inputs, groundtruths



_smpl = None
def get_smpl(args):
    global _smpl
    if _smpl is None:
        _smpl = build_body_model(args, device='cpu')

    return _smpl


def get_smpl_output(target):
    _smpl_output = _smpl(
        body_pose=target['pose'][:, 1:],
        betas=target['betas'],
        global_orient=target['pose'][:, :1],
        transl=target['transl'],
        pose2rot=False)
    return _smpl_output


def get_keypoints_noise(joint_type, noise_std):
    if joint_type == 'J17':
        noise_factor = torch.tensor([2, 3, 5, 5, 3, 2, 2, 3, 3, 3, 3, 2, 6, 2,
                                        6, 7, 2]).unsqueeze(-1).expand(17, 3)
    elif joint_type == 'OP19':
        noise_factor = torch.tensor([2, 3, 5, 5, 3, 2, 2, 3, 3, 3, 3, 2, 6,
                                        6, 1, 1, 1, 1, 1]).unsqueeze(-1).expand(19, 3)
    elif joint_type == 'TC16':
        noise_factor = torch.tensor([2, 5, 3, 2, 5, 3, 2, 5, 4, 3, 5, 3, 2,
                                        5, 3, 2]).unsqueeze(-1).expand(16, 3)
        peak_factor = torch.tensor([0, 1, 5, 5, 1, 5, 5, 0.2, 0.5, 1, 2, 5,
                                    3, 2, 5, 3]).unsqueeze(-1).expand(16, 3)
    noise_factor = noise_factor.float() * noise_std

    return noise_factor, peak_factor


def get_keypoints(target):
    """Get 3D keypoints of the motion"""
    smpl_output = get_smpl_output(target)
    keypoints = smpl_output.joints
    target['gt_keypoints'] = keypoints
    return target


def get_synthetic_IMU(target):
    """Get synthetic IMU data of the motion"""
    smpl_output = get_smpl_output(target)
    ori = smpl_pose_to_imu_orientation(target['pose'], _smpl.parents,
                                        target['calib'])
    gyr = smpl_pose_to_angular_velocity(ori, _smpl.parents)
    acc = smpl_pose_to_acceleration(smpl_output.vertices, ori,
                                    _smpl.parents)
    target['gyr'] = gyr
    target['acc'] = acc
    return target


def get_calibration_matrix(target, calib_R, noise_std, training=True):
    """Currently only implemented random noise for calibration:
    TODO: Update code to rotate more in axial direction
    """
    noise_A = torch.normal(mean=torch.zeros((calib_R.shape[0], 3)),
                            std=torch.ones((calib_R.shape[0], 3)) * noise_std)
    _calib_R = calib_R.clone()
    if training:
        # Specific sensors have higher error in the placement
        for idx, key in enumerate(_C.SENSOR_LIST):
            if key in ['L_UpArm', 'L_LowArm', 'R_UpArm', 'R_LowArm',
                        'L_UpLeg', 'L_LowLeg', 'R_UpLeg', 'R_LowLeg',
                        'L_Foot', 'R_Foot']:
                noise_A[idx, 0] *= 3
            if key in ['L_UpArm', 'R_UpArm', 'L_Foot', 'R_Foot']:
                noise_A[idx, [1, 2]] *= 3

    noise_R = angle_axis_to_rotation_matrix(noise_A, return_hom=False)
    _calib_R = _calib_R @ noise_R
    target['calib'] = _calib_R
    return target


def augment_smpl_params(target):
    """Agument SMPL parameters
    Type 1. Random rotation
    Type 2. Random translation
    Type 3. Random shape
    """
    seq_length = target['pose'].shape[0]
    angle = torch.rand(1) * 2 * np.pi
    euler = torch.tensor([0, angle, 0]).float().unsqueeze(0)
    rmat = angle_axis_to_rotation_matrix(euler, return_hom=False)
    shape_noise = torch.normal(mean=torch.zeros((1, 10)),
                    std=torch.ones((1, 10))).expand(seq_length, 10)
    target['betas'] = target['betas'] + shape_noise
    target['pose'][:, 0] = rmat @ target['pose'][:, 0]
    target['transl'] = (rmat @ target['transl'].T).squeeze().T
    return target


def augment_keypoints(keypoints, joint_type, noise_std, seq_length):
    """Augment 3D keypoints"""

    noise_factor, peak_factor = get_keypoints_noise(joint_type, noise_std)

    # Noise type 1. Time invariant bias (maybe due to cloth)
    t_invariant_noise = torch.normal(
        mean=torch.zeros((len(noise_factor), 3)), std=noise_factor
    ).unsqueeze(0).expand(seq_length, len(noise_factor), 3)

    # Noise type 2. High frequency jittering noise
    t_variant_noise = torch.normal(
        mean=torch.zeros((seq_length, len(noise_factor), 3)),
        std=noise_factor.unsqueeze(0).expand(
            seq_length, len(noise_factor), 3)/3)

    # Noise type 3. Low frequency high magnitude noise
    peak_noise_mask = (torch.rand(seq_length, noise_factor.size(0)) < 2e-2
                        ).float().unsqueeze(-1).repeat(1, 1, 3)
    peak_noise = peak_noise_mask * torch.randn(3) * noise_std
    peak_noise = peak_noise * peak_factor

    # Calculate loss and confidence (higher loss, less confident)
    noise = t_invariant_noise + t_variant_noise + peak_noise
    conf_randomizer = torch.rand(*noise.shape) * 3 + 15
    conf = torch.exp(-torch.abs(noise)*conf_randomizer).mean(-1, keepdims=True)

    keypoints += noise
    keypoints = torch.cat((keypoints, conf), dim=-1)

    if joint_type == 'OP19':
        keypoints[:, _C.PELVIS_IDX[joint_type]] = keypoints[:, [2, 3]].mean(1)

    return keypoints


def get_glob_smpl_pose_with_augmentation(target, seq_length):
    pose = target['pose'].clone()
    glob_pose = smpl_pose_to_global_pose(pose, _smpl.parents)

    t_invariant_noise = torch.normal(mean=torch.zeros((1, 24, 3)),
            std=torch.ones((1, 24, 3))*1e-2).expand(seq_length, -1, -1)
    t_variant_noise = torch.normal(mean=torch.zeros((seq_length, 24, 3)),
                        std=torch.ones((seq_length, 24, 3))*5e-3)

    noise_A = t_invariant_noise + t_variant_noise
    noise_R = angle_axis_to_rotation_matrix(noise_A.view(-1, 3),
                    return_hom=False).reshape(seq_length, 24, 3, 3)
    glob_pose = glob_pose @ noise_R
    target['glob_pose'] = glob_pose
    return target
