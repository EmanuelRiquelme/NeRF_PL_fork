import torch
from collections import defaultdict
import numpy as np
import mcubes
import trimesh
import argparse
import json
from models.rendering import *
from models.nerf import *

from datasets import dataset_dict

from utils import load_ckpt


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='Generate_poses_Colmap/points',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 600],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--x_axis', nargs="+", type=float, default=[-1, 1],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--y_axis', nargs="+", type=float, default=[-1, 1],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--z_axis', nargs="+", type=float, default=[-1, 1],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--N',type = int,default = 256)
    parser.add_argument('--sigma',type = float,default = 30.)
    parser.add_argument('--scene_name',type = str,default = "NeRF_scene")
    parser.add_argument('--ckpt_path',type = str, default = "ckpts/exp/epoch=X.ckpt")
    return parser.parse_args()

def plot_mesh(params):
    xmin, xmax = params.x_axis[0],params.x_axis[1]
    ymin, ymax = params.y_axis[0],params.y_axis[1]
    zmin, zmax = params.z_axis[0],params.z_axis[1]
    sigma_threshold = params.sigma
    img_wh = params.img_wh
    dataset_name = params.dataset_name
    scene_name = params.scene_name
    root_dir = params.root_dir
    ckpt_path = params.ckpt_path
    N = params.N
    kwargs = {'root_dir': root_dir,
            'img_wh': img_wh}
    if dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
        
    chunk = 1024*32
    dataset = dataset_dict[dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)

    nerf_fine = NeRF()
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval();


    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()

    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i+chunk]) # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded)]
        rgbsigma = torch.cat(out_chunks, 0)
        
    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)

    # The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
    mesh = trimesh.Trimesh(vertices/N, triangles)
    mesh.show()

def save_parameters(params):
    config = {
        "x_axis" : params.x_axis,
        "y_axis" : params.y_axis,
        "z_axis" : params.z_axis,
        "sigma_threshold" : params.sigma,
        "img_wh" : params.img_wh,
        "dataset_name" : params.dataset_name,
        "scene_name" : params.scene_name,
        "root_dir" : params.root_dir,
        "ckpt_path" : params.ckpt_path,
    }
    with open("mesh_conf.json", "w") as outfile:
        json.dump(config,outfile)

    print("parameters saved!")


if __name__=='__main__':
    params =get_opts()
    assert params.ckpt_path == 'ckpts/exp/epoch=X.ckpt' \
    "Indicate the NeRF model location with the --ckpt_path argument."
    plot_mesh(params)
    save_parameters(params)    
