import torch
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows
import numpy as np

import joblib
from tqdm import tqdm
import os
import os.path as osp
from pdb import set_trace as st

from ..models.smpl import build_body_model
import lib.data.data_utils as d_utils
from lib.utils.smpl_utils import (smpl_pose_to_acceleration,
                                  smpl_pose_to_imu_orientation,
                                  smpl_pose_to_angular_velocity,
                                  smpl_pose_to_global_pose)
from lib.utils.conversion import angle_axis_to_rotation_matrix
from configs import constants as _C



class AMASS(Dataset):
    def __init__(self, args, label_pth, is_train, **kwargs):
        super(AMASS, self).__init__()

        self.is_train = is_train
        self.model_type = args.model_type
        self.labels = joblib.load(label_pth)
        self.seq_length = args.seq_length
        self.joint_type = args.joint_type

        self.prepare_video_batch()
        print(f'AMASS dataset number of videos: {len(self.video_indices)}')

        # Initialization
        self.basic_R = torch.Tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
                                    ).unsqueeze(0).float()
        d_utils.get_smpl(args)
        calib_R = np.load(_C.CALIB_FILE, allow_pickle=True).item()
        self.calib_R = []
        for key in _C.SENSOR_LIST:
            self.calib_R.append(torch.from_numpy(calib_R[key]).float())
        self.calib_R = torch.stack(self.calib_R)
        self.reset_calib_noise(std=(args.calib_noise * self.is_train))
        self.reset_keypoints_noise(std=(args.keypoints_noise * self.is_train))

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def prepare_video_batch(self, step_size=None):
        self.video_indices = []
        video_names_unique, group = np.unique(
            self.labels['vid_name'], return_index=True)
        perm = np.argsort(group)
        self.video_names_unique, self.group = video_names_unique[perm], group[perm]
        indices = np.split(np.arange(0, self.labels['vid_name'].shape[0]), self.group[1:])

        if step_size is None:
            if self.is_train:
                step_size = self.seq_length // 5
            else:
                step_size = self.seq_length

        for idx in range(len(video_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.seq_length:
                continue
            chunks = view_as_windows(
                indexes, (self.seq_length), step=step_size)
            if self.is_train:
                chunks = chunks[np.random.randint(5)::5]
            start_finish = chunks[:, (0, -1)].tolist()
            self.video_indices += start_finish

    def reset_calib_noise(self, std=1.5e-1):
        self.calib_noise = std

    def reset_keypoints_noise(self, std=5e-3):
        self.keypoints_noise = std

    def get_single_item(self, index):
        start_index, end_index = self.video_indices[index]
        pose = self.labels['pose'][start_index:end_index+1]
        pose = torch.from_numpy(pose).reshape(-1, 3).float()
        pose = angle_axis_to_rotation_matrix(
            pose, return_hom=False).reshape(-1, 24, 3, 3)
        pose[:, 0] = self.basic_R @ pose[:, 0]
        betas = self.labels['betas'][start_index:end_index+1]
        transl = self.labels['transl'][start_index:end_index+1]
        transl = (self.basic_R @ (torch.from_numpy(
            transl).T.float())).squeeze().T
        target = {'pose': pose.float(),
                  'betas': torch.from_numpy(betas).float(),
                  'transl': transl.float(),
                  'vid_name': self.labels['vid_name'][start_index]}

        if (not self.is_train) and self.joint_type == 'TC16':
            target = self.get_real_keypoints(start_index, end_index, target)
            if self.keypoints_noise > 0:
                target['input_keypoints'] = d_utils.augment_keypoints(
                    target['input_keypoints'].clone()[..., :3], self.joint_type,
                    self.keypoints_noise, self.seq_length)
        else:
            if self.is_train:
                target = d_utils.augment_smpl_params(target)
            target = d_utils.get_glob_smpl_pose_with_augmentation(
                target, self.seq_length)
            target = d_utils.get_keypoints(target)
            target['input_keypoints'] = d_utils.augment_keypoints(
                target['gt_keypoints'].clone(), self.joint_type,
                self.keypoints_noise, self.seq_length)

        if self.model_type in ['fusion', 'imu']:
            target = d_utils.get_calibration_matrix(
                target, self.calib_R, self.calib_noise, self.is_train)
            if self.is_train:
                target = d_utils.get_synthetic_IMU(target)
            else:
                target = self.get_real_IMU(start_index, end_index, target)
        return target

    def get_real_keypoints(self, start_index, end_index, target):
        """Get Predicted and Ground-Truth keypoints for Total Capture Data"""
        pred_keypoints = self.labels['pred_joints'][start_index:end_index+1] / 1e3
        pred_keypoints[..., -1] *= 1e3
        gt_keypoints = self.labels['gt_joints'][start_index:end_index+1] / 1e3
        target['gt_keypoints'] = torch.from_numpy(gt_keypoints).float()
        target['input_keypoints'] = torch.from_numpy(pred_keypoints).float()
        return target

    def get_real_IMU(self, start_index, end_index, target):
        """Get Real IMU data for testing set"""
        _ori = self.labels['ori'][start_index:end_index+1]
        _acc = self.labels['acc'][start_index:end_index+1]
        _gyr = self.labels['gyr'][start_index:end_index+1]
        ori, acc, gyr = [], [], []

        for sensor in _C.SENSOR_LIST:
            sensor_idx = list(_C.SENSOR_TO_SMPL_MAP.keys()).index(sensor)
            ori.append(_ori[:, sensor_idx])
            acc.append(_acc[:, sensor_idx])
            gyr.append(_gyr[:, sensor_idx])

        s_map = list(_C.SENSOR_TO_SMPL_MAP.keys())
        ori = np.stack([ori[s_map.index(s)] for s in _C.SENSOR_LIST])
        acc = np.stack([acc[s_map.index(s)] for s in _C.SENSOR_LIST])
        gyr = np.stack([gyr[s_map.index(s)] for s in _C.SENSOR_LIST])
        target['ori'] = torch.from_numpy(ori).float().transpose(1, 0)
        target['acc'] = torch.from_numpy(acc).float().transpose(1, 0) / 9.81
        target['gyr'] = torch.from_numpy(gyr).float().transpose(1, 0) / 10

        if self.calib_noise > 0:
            for idx in range(len(_C.SENSOR_LIST)):
                target['acc'][:, idx] = target['acc'][:, idx] @ self.calib_R[idx].T @ target['calib'][idx]
                target['gyr'][:, idx] = target['gyr'][:, idx] @ self.calib_R[idx].T @ target['calib'][idx]

        return target


def setup_amass_dloader(args, eval_only=False, **kwargs):

    print('Load AMASS Dataset...')

    test_dataset = AMASS(args,
                         label_pth=_C.AMASS_TEST_LABEL_PTH,
                         is_train=False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory,
                                collate_fn=d_utils.make_collate_fn())

    if eval_only:
        return None, test_dataloader

    train_dataset = AMASS(args,
                          label_pth=_C.AMASS_TRAIN_LABEL_PTH,
                          is_train=True)
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=args.pin_memory,
                                 collate_fn=d_utils.make_collate_fn())

    return train_dataloader, test_dataloader
