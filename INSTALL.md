This repo implements our ICCV paper "Cross View Fusion for 3D Human Pose Estimation"
https://chunyuwang.netlify.com/img/ICCV_Cross_view_camera_ready.pdf


# Quick start
## Installation
1. Clone this repo, and we'll call the directory that you cloned multiview-pose as ${POSE_ROOT}
2. Install dependencies.
3. Download pytorch imagenet pretrained models. Please download them under ${POSE_ROOT}/models, and make them look like this:

   ```
   ${POSE_ROOT}/models
   └── pytorch
       └── imagenet
           ├── resnet152-b121ed2d.pth
           ├── resnet50-19c8e357.pth
           └── mobilenet_v2.pth.tar
   ```
   They can be downloaded from the following link:
   https://onedrive.live.com/?authkey=%21AF9rKCBVlJ3Qzo8&id=93774C670BD4F835%21930&cid=93774C670BD4F835
   
   

4. Init output(training model output directory) and log(tensorboard log directory) directory.

   ```
   mkdir ouput 
   mkdir log
   ```

   and your directory tree should like this

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments-local
   ├── experiments-philly
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   ├── requirements.txt
   ```

## Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/), the original annotation files are matlab's format. We have converted to json format, you also need download them from [OneDrive](https://1drv.ms/u/s!AjX41AtnTHeTiE3HX1HCYK2QcE0V?e=l9c3lb).
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- MPII
    |-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   |-- valid.json
        |-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

If you zip the image files into a single zip file, you should organize the data like this:

```
${POSE_ROOT}
|-- data
`-- |-- MPII
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images.zip
            `-- images
                |-- 000001163.jpg
                |-- 000003072.jpg
```



**For Human36M data**, please follow https://github.com/CHUNYUWANG/H36M-Toolbox to prepare images and annotations, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- h36m
    |-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        |-- images
            |-- s_01_act_02_subact_01_ca_01 
            |-- s_01_act_02_subact_01_ca_02
```

If you zip the image files into a single zip file, you should organize the data like this:
```
${POSE_ROOT}
|-- data
`-- |-- h36m
    `-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        `-- images.zip
            `-- images
                |-- s_01_act_02_subact_01_ca_01
                |-- s_01_act_02_subact_01_ca_02
```


**Limb length prior for 3D Pose Estimation**, please download the limb length prior data from 
https://1drv.ms/u/s!AjX41AtnTHeTiQt7xLFmMLrcBuYq?e=KSia8L

put it in data/pict/pairwise.pkl


## 2D Training and Testing
**Multiview Training on Mixed Dataset (MPII+H36M) and testing on H36M**
```
python run/pose2d/train.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml
python run/pose2d/valid.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml
```
## 3D Testing
**Multiview testing on H36M (based on CPU or GPU)**
```
python run/pose3d/estimate.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml (CPU Version)
python run/pose3d/estimate_cuda.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml (GPU Version)
```
### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{multiviewpose,
    author={Qiu, Haibo and Wang, Chunyu and Wang, Jingdong and Wang, Naiyan and Zeng, Wenjun},
    title={Cross View Fusion for 3D Human Pose Estimation},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```

### Acknowledgement
This repo uses a lot of code from the following repo:
https://github.com/microsoft/human-pose-estimation.pytorch

We would like to thank the authors for publishing their code.

