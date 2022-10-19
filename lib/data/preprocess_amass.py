# Reference: https://github.com/mkocabas/VIBE
from utils.conversion import angle_axis_to_rotation_matrix

import os; import os.path as osp
import json
import joblib
import torch
import numpy as np
from tqdm import tqdm


amass_fldr = 'data/AMASS'
_, all_seqs, _ = next(os.walk(amass_fldr))
out_fname = 'amass_db.pt'

amass_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'transl']
amass_joint_mapper = np.arange(24); amass_joint_mapper[-1] = 37
amass_joint_mapper = np.arange(0, 150).reshape((-1, 3))[amass_joint_mapper].reshape(-1)

target_fps = 25

def read_data(seqs=None):
    if seqs is None:
        seqs = all_seqs

    out_dict = {'pose': [], 'betas': [], 'transl': [], 'vid_name': []}

    pbar_seq = tqdm(seqs, desc='Generating Data ...')
    for seq in seqs:
        vid_names = []
        poses, betas, transls = [], [], []

        seq_fldr = osp.join(amass_fldr, seq)
        subjs = os.listdir(seq_fldr)

        pbar_subj = tqdm(subjs, desc=f'Data: {seq}', leave=False)
        for subj in subjs:
            acts = [x for x in os.listdir(osp.join(seq_fldr, subj)) if x.endswith('.npz')]

            pbar_act = tqdm(acts, desc=f'Subject: {subj}', leave=False)
            for act in acts:
                fname = osp.join(seq_fldr, subj, act)
                if fname.endswith('shape.npz'):
                    continue

                data = np.load(fname)
                mocap_framerate = int(data['mocap_framerate'])
                retain_freq = mocap_framerate // target_fps
                pose = data['poses'][::retain_freq, amass_joint_mapper]
                transl = data['trans'][::retain_freq]

                if pose.shape[0] < 60:
                    continue

                shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
                vid_name = np.array([f'{seq}_{subj}_{act[:-4]}']*pose.shape[0])

                vid_names.append(vid_name)
                poses.append(pose)
                betas.append(shape)
                transls.append(transl)

                pbar_act.update(1)
            pbar_subj.update(1)

        poses = np.concatenate(poses, axis=0)
        betas = np.concatenate(betas, axis=0)
        transls = np.concatenate(transls, axis=0)
        vid_names = np.concatenate(vid_names, axis=0)
        seq_list = np.array([seq] * poses.shape[0])
        out_dict['pose'].append(poses)
        out_dict['betas'].append(betas)
        out_dict['transl'].append(transls)
        out_dict['vid_name'].append(vid_names)

        pbar_seq.update(1)

    out_dict['pose'] = np.concatenate(out_dict['pose'], axis=0)
    out_dict['betas'] = np.concatenate(out_dict['betas'], axis=0)
    out_dict['transl'] = np.concatenate(out_dict['transl'], axis=0)
    out_dict['vid_name'] = np.concatenate(out_dict['vid_name'], axis=0)

    return out_dict


if __name__ == '__main__':
    data = read_data()
    joblib.dump(data, out_fname)
