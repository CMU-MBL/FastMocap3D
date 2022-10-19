import torch
import numpy as np

import smplx
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX
from smplx.utils import SMPLOutput as ModelOutput
from smplx.lbs import vertices2joints

from einops import rearrange
from lib.utils.conversion import rot6d_to_rotmat
from configs import constants as _C


# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
}


JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow',
'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist',
'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle',
'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye',
'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe',
'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, joint_regressor=None, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        if joint_regressor is not None:
            J_regressor_extra = np.load(joint_regressor)
            self.register_buffer('J_regressor_extra', torch.tensor(
                J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.register_buffer(
            'wrist', torch.eye(3)[None, None].expand(self.batch_size, 2, -1, -1))

    def get_output(self, smpl_params, groundtruths=None):
        b = self.batch_size
        pred_pose, pred_betas, pred_transl = smpl_params
        pred_global_orient = pred_pose[:, :, :1].view(b, 1, 3, 3)
        pred_pose= pred_pose[:, :, 1:].view(b, 21, 3, 3)
        pred_pose_w_hand = torch.cat((pred_pose, self.wrist), dim=1)

        if pred_betas is None:
            pred_betas = groundtruths['betas']
            pred_transl = groundtruths['transl']
            pred_global_orient = groundtruths['pose'][:, :, :1]

        output = self.forward(body_pose=pred_pose_w_hand,
                              global_orient=pred_global_orient,
                              betas=pred_betas.view(b, 10),
                              transl=pred_transl.view(b, 3),
                              pose2rot=False)

        return output

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if self.J_regressor_extra is not None:
            joints = vertices2joints(self.J_regressor_extra,
                                     smpl_output.vertices)
        else:
            joints = smpl_output.joints[:, self.joint_map, :]

        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

    def get_joints(self, cat_org_joints=False, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if self.J_regressor_extra is not None:
            extra_joints = vertices2joints(self.J_regressor_extra,
                                           smpl_output.vertices)
            if cat_org_joints:
                joints = torch.cat((smpl_output.joints, extra_joints), dim=1)
            else:
                joints = extra_joints
        else:
            joints = smpl_output.joints[:, self.joint_map, :]

        return joints


def build_body_model(args, device=None, batch_size=None, **kwargs):

    if batch_size is None: batch_size = args.batch_size * args.seq_length
    if device is None: device = args.device
    body_model = SMPL(
        joint_regressor=_C.SMPL_REGRESSOR[args.joint_type],
        model_path=_C.SMPL_FLDR,
        gender=args.sex,
        batch_size=batch_size,
        create_transl=False
    ).to(device)
    return body_model
