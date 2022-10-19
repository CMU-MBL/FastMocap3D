from configs import constants as _C
from lib.utils.conversion import rotation_matrix_to_angle_axis

import torch
import numpy as np


def smpl_pose_to_global_pose(pose, parents):
    """Recover global orientation of body joints using kinematic tree

    Args:
        pose: pose parameters of SMPL model (include global orientation)
        parents: kinematic tree defined by SMPL

    Return:
        pose in global coordinate system
    """

    org_shape = pose.shape
    pose = pose.reshape(-1, *org_shape[-3:])
    results = []
    root_pose = pose[:, 0]
    results += [root_pose]
    for i in range(1, pose.size(1)):
        loc_pose = pose[:, i]
        global_pose = results[parents[i]] @ loc_pose
        results.append(global_pose)

    results = [result.unsqueeze(1) for result in results]
    results = torch.cat(results, dim=1)
    results = results.reshape(*org_shape)
    return results


def global_pose_to_smpl_pose(global_pose, parents):
    """Recover SMPL local pose of body joints using kinematic tree

    Args:
        global_pose: given global pose of SMPL joints
        parents: kinematic tree defined by SMPL

    Return:
        pose in local coordinate system
    """

    org_shape = global_pose.shape
    global_pose = global_pose.reshape(-1, *org_shape[-3:])
    results = []
    root_pose = global_pose[:, 0]
    results += [root_pose]
    for i in range(1, global_pose.size(1)):
        curr_pose = global_pose[:, i]
        loc_pose = global_pose[:, parents[i]].transpose(1, 2) @ curr_pose
        results += [loc_pose]

    results = [result.unsqueeze(1) for result in results]
    results = torch.cat(results, dim=1)
    results = results.reshape(*org_shape)
    return results


def smpl_pose_to_imu_orientation(pose, parents, calib_R):
    """Calculate synthetic sensor oirentation data

    Args:
        pose: SMPL pose parameters
        parents: SMPL kinematic tree
        calib_R: SMPL to sensor coordinate calibration

    Return:
        ori: Synthetic orientationt data
    """
    device = pose.device
    dtype = pose.dtype
    pose = pose.view(-1, 24, 3, 3)
    glob_pose = smpl_pose_to_global_pose(pose, parents)
    results = []
    for sensor in _C.SENSOR_LIST:
        sensor_idx = list(_C.SENSOR_TO_SMPL_MAP.keys()).index(sensor)
        joint = _C.SENSOR_TO_SMPL_MAP[sensor]
        SMPL_idx = _C.SMPL_JOINT_NAMES.index(joint)
        tmp = glob_pose[:, SMPL_idx] @ calib_R[sensor_idx]
        results.append(tmp.unsqueeze(1))

    ori = torch.cat(results, dim=1).to(device=device, dtype=dtype)
    return ori


def smpl_pose_to_acceleration(vertices, ori, parents):
    """Calculate synthetic accelerometer data

    Args:
        vertices: SMPL vertices location
        pose: SMPL pose parameters
        parents: SMPL kinematic tree

    Return:
        acc: Synthetic accelerometer data
    """
    device = vertices.device
    dtype = vertices.dtype
    results = []
    for idx, sensor in enumerate(_C.SENSOR_LIST):
        loc = vertices[:, _C.SENSOR_TO_VERTS[sensor]]
        tmp = (loc[2:] + loc[:-2] - 2 * loc[1:-1]) * (_C.AMASS_FPS ** 2)
        tmp[:, 1] += 9.81
        tmp = (tmp.unsqueeze(1) @ ori[1:-1, idx]).squeeze()
        results.append(tmp.unsqueeze(1))
    acc_ = torch.cat(results, dim=1)
    acc = torch.zeros((acc_.shape[0] + 2, acc_.shape[1], 3)).float().to(
        device=device, dtype=dtype)
    acc[1:-1] = acc_
    acc = acc/9.81
    return acc

def smpl_pose_to_angular_velocity(ori, parents=None):
    """Calculate synthetic gyroscope data

    Args:
        pose: SMPL pose parameters
        parents: SMPL kinematic tree

    Return:
        gyr: Synthetic gyroscope (angular velocity) data
    """
    device = ori.device
    dtype = ori.dtype
    results = []
    for idx, sensor in enumerate(_C.SENSOR_LIST):
        tmp = torch.transpose(ori[:-1, idx], 1, 2) @ ori[1:, idx]
        tmp = rotation_matrix_to_angle_axis(tmp) * _C.AMASS_FPS
        results.append(tmp.unsqueeze(1))
    gyr_ = torch.cat(results, dim=1)
    gyr = torch.zeros((gyr_.shape[0] + 1, gyr_.shape[1], 3)).float().to(
        device=device, dtype=dtype)
    gyr[:-1] = gyr_
    gyr = gyr/10
    return gyr
