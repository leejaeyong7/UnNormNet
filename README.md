## Visual Representation Learning via Unsupervised Learning of Surface Normal
This repo covers the implementation for ECE 598 final project. This project aims to learn visual representations in a self-supervised manner through unsupervised learning of surface noraml.

Large part of this code is from [Contrastive Multiview Coding (CMC)](https://github.com/HobbitLong/CMC/).


## Installation
This repo was tested with Python 3.7, PyTorch 0.4.1, and CUDA 10.0.  To install: 

```bash
conda create --name unnorm -y
conda activate unnorm
conda install pip opencv matplotlib numpy pyyaml
conda install pytorch=0.4.1 cuda100 torchvision -c pytorch
pip install imgaug tensorboard_logger

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

**Note:** It seems to us that training with Pytorch version >= 1.0 yields slightly worse results. If you find the similar discrepancy and figure out the problem, please report this since we are trying to fix it as well.


## Dataset
We assume that your symlinked `data/imagenet` directory has the following structure:
```
|_ train
|  |_ n01440764/
|  |_ ...
|  |_ n15075141/
|_ val
|  |_ n02095314/
|  |_ ...
|  |_ n04120489/
```
**How-to:** check `dataset/imagenet_cmd.sh` for the script to creat above structure from raw downloaded files.

## Baselines
- [Unofficial] MoCo: Momentum Contrast for Unsupervised Visual Representation Learning ([Paper](https://arxiv.org/abs/1911.05722)) 
- [Unofficial] InsDis: Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination ([Paper](https://arxiv.org/abs/1805.01978))
- [Official] CMC: Contrastive Multiview Coding ([Paper](http://arxiv.org/abs/1906.05849))

## Training on ImageNet100 subset

We compare to `InsDis`, `MoCo`, and `CMC` on a `ImageNet100` subset (but the code allows one to train on full ImageNet simply by setting the flag `--dataset imagenet`):

The pre-training stage:

- For InsDis:
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco_ins.py \
     --batch_size 128 --num_workers 24 --nce_k 16384 --softmax
    ```
- For MoCo:
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco_ins.py \
     --batch_size 128 --num_workers 24 --nce_k 16384 --softmax --moco
    ```
  
The linear evaluation stage:
- For both InsDis and MoCo (lr=10 is better than 30 on this subset, for full imagenet please switch to 30):
    ```
    CUDA_VISIBLE_DEVICES=0 python eval_moco_ins.py --model resnet50 \
     --model_path /path/to/model --num_workers 24 --learning_rate 10
    ```
  
The comparison of `InsDIS`, `MoCo`, `CMC (using YCbCr)`, and `UnNorm (ours)` and on `ImageNet100` subset, is tabulated as below:

|          |Arch | #Params(M) | Loss  | #Negative  | Accuracy |
|----------|:----:|:---:|:---:|:---:|:---:|
|  InsDis | ResNet50 | 24 | NCE  | 16384  |  53.5  |
|  InsDis | ResNet50 | 24 | Softmax-CE  | 16384  |  69.1  |
|  MoCo   | ResNet50 | 24 | NCE  | 16384  |  11.6  |
|  MoCo   | ResNet50 | 24 | Softmax-CE  | 16384  |  73.4  |
|  CMC    | 2xResNet50half | 12 | NCE  | 4096  |  74.5  |
|  CMC    | 2xResNet50half | 12 | Softmax-CE  | 4096  |  75.8  |
|  InsDis+HomoAug | ResNet50 | 24 | Softmax-CE  | 16384  |  |
|  MoCo+HomoAug   | ResNet50 | 24 | Softmax-CE  | 16384  |  |
|  UnNorm | ResNet50 | 24 | - | -  | TODO |
|  UnNorm | 2xResNet50half | 12 | - | - | TODO |


## Training on Full ImageNet

**Note:** For AlexNet, we split across the channel dimension and use each half to encode L and ab. For ResNets, we use a standard ResNet model to encode each view.

NCE flags:
- `--nce_k`: number of negatives to contrast for each positive. Default: 4096
- `--nce_m`: the momentum for dynamically updating the memory. Default: 0.5
- `--nce_t`: temperature that modulates the distribution. Default: 0.07 for ImageNet, 0.1 for STL-10

Path flags:
- `--data_folder`: specify the ImageNet data folder.
- `--model_path`: specify the path to save model.
- `--tb_path`: specify where to save tensorboard monitoring events.

Model flag:
- `--model`: specify which model to use, including *alexnet*, *resnets18*, *resnets50*, and *resnets101*

An example of command line for training CMC (Default: `AlexNet` on Single GPU)
```
CUDA_VISIBLE_DEVICES=0 python train_CMC.py --batch_size 256 --num_workers 36 \
 --data_folder /path/to/data 
 --model_path /path/to/save 
 --tb_path /path/to/tensorboard
```

Training CMC with ResNets requires at least 4 GPUs, the command of using `resnet50v1` looks like
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CMC.py --model resnet50v1 --batch_size 128 --num_workers 24
 --data_folder path/to/data \
 --model_path path/to/save \
 --tb_path path/to/tensorboard \
```

To support mixed precision training, simply append the flag `--amp`, which, however is likely to harm the downstream classification. I measure it on ImageNet100 subset and the gap is about 0.5-1%.

By default, the training scripts will use L and ab as two views for contrasting. You can switch to `YCbCr` by specifying `--view YCbCr`, which yields better results (about 0.5-1%). If you want to use other color spaces as different views, follow the line [here](https://github.com/HobbitLong/CMC/blob/master/train_CMC.py#L146) and other color transfer functions are already available in `dataset.py`.

## Training Linear Classifier

Path flags:
- `--data_folder`: specify the ImageNet data folder. Should be the same as above.
- `--save_path`: specify the path to save the linear classifier.
- `--tb_path`: specify where to save tensorboard events monitoring linear classifier training.

Model flag `--model` is similar as above and should be specified.

Specify the checkpoint that you want to evaluate with `--model_path` flag, this path should directly point to the `.pth` file.

This repo provides 3 ways to train the linear classifier: *single GPU*, *data parallel*, and *distributed data parallel*.

An example of command line for evaluating, say `./models/alexnet.pth`, should look like:
```
CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --dataset imagenet \
 --data_folder /path/to/data \
 --save_path /path/to/save \
 --tb_path /path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --model alexnet --learning_rate 0.1 --layer 5
```

**Note:** When training linear classifiers on top of ResNets, it's important to use large learning rate, e.g., 30~50. Specifically, change `--learning_rate 0.1 --layer 5` to `--learning_rate 30 --layer 6` for `resnet50v1` and `resnet50v2`, to `--learning_rate 50 --layer 6` for `resnet50v3`.

## Pretrained Models
Pretrained weights can be found in [(TODO)](?).


## Contact
For any questions, please contact Jason Ren (zr5@illinois.edu).