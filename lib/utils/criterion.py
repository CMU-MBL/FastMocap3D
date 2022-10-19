import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from configs import constants as _C
from lib.utils.conversion import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.utils.smpl_utils import smpl_pose_to_global_pose
from lib.utils.pose_utils import compute_similarity_transform_batch

from pdb import set_trace as st



class Criterion():
    def __init__(self, args):

        self.lw_pose = args.lw_pose
        self.lw_betas = args.lw_betas
        self.lw_transl = args.lw_transl
        self.lw_keypoints = args.lw_keypoints

        self.joint_type = args.joint_type


    def __call__(self, pred_smpl, pred_keypoints, groundtruths):
        loss_dict = {}

        pred_pose, pred_betas, pred_transl = pred_smpl
        pose, betas, transl, keypoints = [groundtruths[k] for k in ['pose', 'betas', 'transl', 'keypoints']]

        if self.lw_pose > 0 and pred_pose is not None:
            loss_dict = self._get_pose_loss(pose, pred_pose, loss_dict)
        if self.lw_betas > 0 and pred_betas is not None:
            loss_dict = self._get_shape_loss(betas, pred_betas, loss_dict)
        if self.lw_transl > 0 and pred_transl is not None:
            loss_dict = self._get_transl_loss(transl, pred_transl, loss_dict)
        if self.lw_keypoints > 0 and pred_keypoints is not None:
            loss_dict = self._get_keypoints_loss(keypoints, pred_keypoints, loss_dict)

        total_loss = 0
        for key in loss_dict.keys():
            total_loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()

        if pred_keypoints is not None:
            loss_dict['MPJPE'] = self._get_mpjpe(keypoints, pred_keypoints)
        return total_loss, loss_dict


    def _get_pose_loss(self, pose, pred_pose, loss_dict):
        """Calculate SMPL pose loss using rotation matrix space"""

        pose_diff = pose.transpose(-1, -2) @ pred_pose
        pose_diff = torch.abs(rotation_matrix_to_angle_axis(pose_diff.reshape(-1, 3, 3)))
        pose_diff = (pose_diff ** 2).sum(-1).mean()
        loss = pose_diff * self.lw_pose
        loss_dict['Pose Loss'] = loss
        return loss_dict

    def _get_shape_loss(self, betas, pred_betas, loss_dict):
        """Calculate SMPL betas loss"""
        loss = F.mse_loss(betas, pred_betas, reduction='mean') * self.lw_betas
        loss_dict['Shape Loss'] = loss
        return loss_dict

    def _get_transl_loss(self, transl, pred_transl, loss_dict):
        """Calculate SMPL transl loss"""
        loss = F.mse_loss(transl, pred_transl, reduction='mean') * self.lw_transl
        # loss_dict['Transl Loss'] = loss
        return loss_dict

    def _get_keypoints_loss(self, keypoints, pred_keypoints, loss_dict):
        """Calculate keypoints loss"""
        # pelv_idx = _C.PELVIS_IDX[self.joint_type]
        # keypoints = keypoints - keypoints[..., pelv_idx, :].unsqueeze(-2)
        # pred_keypoints = pred_keypoints - pred_keypoints[..., pelv_idx, :].unsqueeze(-2)
        loss = F.mse_loss(keypoints, pred_keypoints, reduction='mean') * self.lw_keypoints
        loss_dict['Keypoints Loss'] = loss
        return loss_dict

    def _get_mpjpe(self, keypoints, pred_keypoints, align=False, return_batch=False):
        pelv_idx = _C.PELVIS_IDX[self.joint_type]
        if align:
            keypoints = keypoints - keypoints[..., pelv_idx:pelv_idx+1, :]
            pred_keypoints = pred_keypoints - pred_keypoints[..., pelv_idx:pelv_idx+1, :]

        jpe = torch.sqrt(torch.sum(torch.square(keypoints - pred_keypoints), -1))
        jpe = jpe.mean((1, 2)) * 1e3
        if return_batch: return jpe.detach()

        mpjpe = jpe.mean().detach()
        return mpjpe.item()

    def _get_pa_mpjpe(self, keypoints, pred_keypoints, return_batch=False):
        b, f, j = keypoints.shape[:3]
        keypoints = keypoints.detach().cpu().numpy().reshape(b * f, j, 3)
        pred_keypoints = pred_keypoints.detach().cpu().numpy().reshape(b * f, j, 3)

        pred_keypoints = compute_similarity_transform_batch(pred_keypoints, keypoints)
        keypoints = keypoints.reshape(b, f, j, 3)
        pred_keypoints = pred_keypoints.reshape(b, f, j, 3)

        jpe = np.sqrt(np.sum(np.square(keypoints - pred_keypoints), -1))
        jpe = jpe.mean((1, 2)) * 1e3
        if return_batch: return jpe

        mpjpe = jpe.mean()
        return mpjpe


    def _get_mpjae(self, pose, pred_pose, full_pose=False, reduction='mean'):
        """ Calculate Mean Per Joint Angle Error
        Args:
            pose: Groundtruth joint angle in 6D format, torch.Tensor (B, F, , 22, 3 3)
            pred_pose: Predicted joint angle in 6D format, torch.Tensor (B, F, 132)
            full_pose: Return joint angle error for all joints (lower limb only if False)

        Return mpjae - Joint angle error
        """

        B, F, J = pose.shape[:3]
        angle = rotation_matrix_to_angle_axis(pose.reshape(-1, 3, 3)).reshape(B, F, J, 3)
        # pred_angle = rotation_matrix_to_angle_axis(rot6d_to_rotmat(pred_pose)).reshape(B, F, J, 3)
        pred_angle = rotation_matrix_to_angle_axis(pred_pose.reshape(-1, 3, 3)).reshape(B, F, J, 3)

        # error in shape B, F, J, 3
        error = torch.abs(angle - pred_angle) * 180 / np.pi

        # retreat uncontinuity issue
        error[error>180] = (360 - error[error>180])

        # Root Mean Square Error
        rmse = torch.sqrt((error ** 2).mean(1))

        if not full_pose:
            rmse = rmse[:, [1, 2, 4, 5, 7, 8]]

        if reduction == 'mean': rmse = rmse.mean((1, 2))
        return rmse
