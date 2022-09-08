
This is an implementation of [NeRF](https://arxiv.org/abs/2003.08934), particularly a fork of an implementation of NeRF that uses Pytorch Lightning.

### What's brings this new fork that the [original implementation does not have](https://github.com/kwea123/nerf_pl):
1-The original repo didn't include the versions of the dependencies, which is a problems because the implementation is quite old and many features are depricated as of today, this repo include the right permutation of versions that make the model work.

2-This implementation does not rely in a [custom Cuda kernel](https://github.com/aliutkus/torchsearchsorted/tree/1e0ffc3e0663ffda318b4e28348efd90313d08f3) to perform the samples instead it uses a feature called searchsorted from torch.

3-It not relient on notebooks to generate the meshes.

### How to use it.
1-Clone the repository.
``` bash
    git clone https://github.com/EmanuelRiquelme/NeRF_PL_fork
```
2-Create a conda environment
``` bash
    conda create --name nerf python=3.6
```
3-Activate the environment
```bash
    conda activate nerf
```
4-Install the dependencies
```bash
    pip3 install -r requirements.txt
```
5-Execute the train.py script to train the model

### How to train the model.
1-First you will need the images and the [poses of the object](https://en.wikipedia.org/wiki/Six_degrees_of_freedom), if you need to generate the poses there's also a [repo](https://github.com/EmanuelRiquelme/Generate_poses_Colmap).

2-Run the train.py script we need to have some arguments in mind:

-**In order to train the model on custom data the parameter dataset_name should be llff**

-**Root_dir is the directory where the poses and the images are stored, if the poses were extracted using the mentioned [repo](https://github.com/EmanuelRiquelme/Generate_poses_Colmap), then the root_dir would be the folder called points.**

-**The resolution option has to comply with the aspect ratio of the origional set of images using the argument(img_wh).**

-**Is possible to retrain a model by placing the model directory in the --ckpt_path option**

-**You can see more info in the opt.py script.**

So a basic way to run the script as an example would be:
``` bash
    python3 train.py --root_dir .../Generate_poses_Colmap/points --dataset_name llff
```
### How to create and save meshes:
1-We need to create the mesh running the evaluate_mesh.py script we need to have some arguments in mind:

-**root_dir img_wh and dataset_name have the same meaning here.**

-**ckpt_path is the location of the .ckpt scene representation model.**

-**scene_name would be the name of the mesh.**

-**The sigma argument reduces noise within the mesh and it's value should be between [0,100].**

-**the Axis cordinates should have a difference of 2 per dimension (the whole objective of this script is to find the best perspective of the object).**

-**At the end of the script a json file would be created for a following script.**

-**You need to run this script many times until have the right location of the object.**

So a basic way to run the script as an example would be:
``` bash
    python3 eval.py \
    --root_dir .../Generate_poses_Colmap/points \
    --dataset_name llff --scene_name NeRF_mesh \
    --ckpt_path ckpts/exp/epoch=X.ckpt 
```
2- Run the script create_mesh.py to create the mesh using the parameters created in the previous step:
-**The argument N determines the quality of the mesh feel free to modify the argument acording to the vram available
-**The format of the mesh would be .vol and it could be open with [meshlab](https://www.meshlab.net/).**

``` bash
    python3 create_mesh.py
```

### Todo
- [ ] add more assertments to the arguments.
