from copy import copy
import os; import os.path as osp
from pdb import set_trace as st

import cv2
import imageio
from PIL import Image, ImageDraw

import torch
import numpy as np
from tqdm import tqdm

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    Materials,
)


__author__ = "Soyong Shin"


calib = np.load('dataset/TC_cam1_calib.npy', allow_pickle=True).item()


class Renderer():
    def __init__(self, device, faces):

        self.device = device
        self.faces = torch.from_numpy((faces).astype('int')).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -20.0]])
        self.create_camera()
        self.create_renderer()


    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""
        self.fx = 5e3
        self.fy = 5e3
        self.height = 720
        self.width = 640
#         self.fx = 1.4e4
#         self.fy = 1.4e4
#         self.height = 1440
#         self.width = 1440

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=(self.height, self.width),
                    blur_radius=0, faces_per_pixel=50,)
            ),
            shader=HardPhongShader(device=self.device, lights=self.lights)
        )

    def create_camera(self):
        # self.R = calib['R'].unsqueeze(0).float().to(self.device)
        self.R = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]).float().to(self.device)
        self.T = torch.tensor([0, 0, 20]).unsqueeze(0).float().to(self.device)
        self.K = torch.tensor([[self.fx, 0, self.width//2],
                               [0, self.fy, self.height//2],
                               [0, 0, 1]]).unsqueeze(0).float().to(self.device)
        self.cameras = PerspectiveCameras(device=self.device, R=self.R, T=self.T,
                                    focal_length=((self.fx, self.fy), ),
                                    principal_point=((self.width//2, self.height//2), ),
                                    image_size=((self.height, self.width), ),
                                    in_ndc=False)

    def render(self, vertices, color, image=None):
        vertices = vertices.unsqueeze(0)
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(color).to(device=self.device)
        mesh = Meshes(verts=vertices, faces=self.faces.unsqueeze(0),
                      textures=TexturesVertex(textures),)
        materials = Materials(
            device=self.device,
            specular_color=[[0.7, 0.7, 0.7]],
            shininess=0
        )
        results = self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights)

        rgb = results[0, ..., :3]
        rgb = torch.flip(rgb, [0, 1]).detach().cpu().numpy()
        depth = results[0, ..., -1]
        depth = torch.flip(depth, [0, 1]).detach().cpu().numpy()

        if image is None:
            image = np.ones_like(rgb)

        image[depth > 1e-3] = (rgb[depth > 1e-3])
        return (image * 255).astype(np.uint8), (depth > 1e-5)

    def render_keypoints(self, keypoints, image=None, linewidth=5):
        image = (image * 255).astype(np.uint8)
        x2d = pose3d_to_pose2d(keypoints, self.R[0], self.T.T, self.K[0])
        x2d = x2d.detach().cpu().numpy().astype('int')
        if x2d.shape[0] == 17: j_type = 'J17'
        elif x2d.shape[0] == 19: j_type = 'OP19'
        elif x2d.shape[0] == 16: j_type = 'TC16'
        connectivity = CONNECTIVITY[j_type]
        color = COLOR[j_type]

        x, y = x2d[:, 0], x2d[:, 1]
        for idx, index_set in enumerate(connectivity):
            xs, ys = [], []
            for index in index_set:
                if (x[index] > 1e-5 and y[index] > 1e-5):
                    xs.append(x[index])
                    ys.append(y[index])
            if len(xs) == 2:
                # Draw line
                start = (xs[0], ys[0])
                end = (xs[1], ys[1])
                image = cv2.line(image, start, end, color[idx], linewidth)

        return image


    def create_video(self, gt_vertices_list, pred_vertices_list, keypoints_list, out_dir, out_name):
        """ Generate comparison video of prediction to groundtruth"""

        gt_color = (0.9, 0.9, 1)
        pred_color = (1.0, 0.9, 0.9)

        if isinstance(gt_vertices_list, list):
            gt_vertices_list = torch.cat(gt_vertices_list, dim=0)
            pred_vertices_list = torch.cat(pred_vertices_list, dim=0)
            keypoints_list = torch.cat(keypoints_list, dim=0)

        os.makedirs(out_dir, exist_ok=True)
        BG = np.ones((self.height, self.width, 3))
        images = []
        prog_bar = tqdm(range(len(gt_vertices_list)), desc=f'Generating {out_name}', leave=False)
        for frame, (gt_vertices, pred_vertices, keypoints) in enumerate(zip(
                gt_vertices_list, pred_vertices_list, keypoints_list)):
            gt_image = self.render(gt_vertices, gt_color, BG.copy())
            input_image = self.render_keypoints(keypoints, BG.copy())
            pred_image = self.render(pred_vertices, pred_color, BG.copy())
            prog_bar.update(1)
            image = np.concatenate((gt_image, input_image, pred_image), axis=1).astype(np.uint8)

            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f'Frame: {(frame + 1):05d}')
            images.append(np.array(image))

        imageio.mimsave(osp.join(out_dir, out_name), images, fps=30)


# # Draw 2D keypoints
CONNECTIVITY = {'J17': [(0,1), (1,2), (5,4), (4,3), (2,14), (3,14), (14,15), (15,12),
                (12,16), (16,13), (6,7), (7,8), (11,10), (10,9), (8,12), (9,12)],
                'OP19': [(0,1), (1,2), (5,4), (4,3), (2,13), (3,13), (13,12),
                (12,14), (6,7), (7,8), (11,10), (10,9), (8,12), (9,12),
                (14,15), (15,17), (14,16), (16,18)],
                'TC16': [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7),
                         (7,8), (8,13), (13,14), (14,15), (8,10), (10,11), (11,12),
                         (8, 9)]}

COLOR = {'J17': [(0, 153, 153), (0, 153, 153),  # right leg
        (0, 0, 153), (0, 0, 153),  # left leg
        (0, 153, 153), (0, 0, 153), # hip
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (51, 153, 0),   # left arm
        (153, 153, 0), (0, 153, 0)],  # shoulder
        'OP19': [(0, 153, 153), (0, 153, 153),  # right leg
        (0, 0, 153), (0, 0, 153),  # left leg
        (0, 153, 153), (0, 0, 153), # hip
        (153, 0, 0),   (153, 0, 102),  # body
        (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (51, 153, 0),   # left arm
        (153, 153, 0), (0, 153, 0),  # shoulder
        (153, 51, 153), (153, 51, 153), (153, 51, 153), (153, 51, 153)],  # Face
        'TC16': [(0, 153, 153), (0, 153, 153), (0, 153, 153),
        (0, 0, 153), (0, 0, 153), (0, 0, 153),
        (153, 0, 0), (153, 0, 0),
        (153, 153, 0), (153, 153, 0), (153, 153, 0),
        (0, 153, 0), (0, 153, 0), (0, 153, 0),
        (153, 51, 153)]
         }


def pose3d_to_pose2d(x3d, _R, _T, _K):
    loc3d = _R @ x3d.T + _T
    loc2d = torch.div(loc3d, loc3d[2])
    x2d = torch.matmul(_K, loc2d)[:2]
    x2d = x2d.T

    return x2d
