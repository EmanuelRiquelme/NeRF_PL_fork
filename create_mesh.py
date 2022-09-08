import json
import numpy as np
import argparse

from datasets import dataset_dict
from models.nerf import *
from utils import load_ckpt

def N_value():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',type = int,default = 256,help='quality of the mesh; reduce if \
        there is no vram left')
    return parser.parse_args()

def create_mesh(file_name = "mesh_conf.json",N = 512):
    params = json.load(open(file_name))
    x_axis = params["x_axis"]
    y_axis = params["y_axis"]
    z_axis = params["z_axis"]
    xmin, xmax = x_axis[0],x_axis[1]
    ymin, ymax = y_axis[0],y_axis[1]
    zmin, zmax = z_axis[0],z_axis[1]

    sigma_threshold = params["sigma_threshold"]
    img_wh = params["img_wh"]
    dataset_name = params["dataset_name"]
    scene_name = params["scene_name"]
    root_dir = params["root_dir"]
    ckpt_path = params["ckpt_path"]

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

    a = 1-np.exp(-(xmax-xmin)/N*sigma)
    a = a.flatten()
    rgb = (rgbsigma[:, :3].cpu().numpy()*255).astype(np.uint32)
    i = np.where(a>0)[0] # valid indices (alpha>0)

    rgb = rgb[i]
    a = a[i]
    s = rgb.dot(np.array([1<<24, 1<<16, 1<<8])) + (a*255).astype(np.uint32)
    res = np.stack([i, s], -1).astype(np.uint32).flatten()
    with open(f'{scene_name}.vol', 'wb') as f:
        f.write(res.tobytes()) 
    print("Mesh created succesfuly!")

if __name__=='__main__':
    N =N_value()
    create_mesh(N = N.N)
