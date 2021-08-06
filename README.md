# TransGAN: Two Pure Transformers Can Make One Strong GAN
Implementation of the paper:

> Yifan Jiang, Shiyu Chang and Zhangyang Wang. [TransGAN: Two Pure Transformers Can Make One
Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074). 

![Architecture](./images/architecture.png)

This repository implements conditional GAN image generation.

See [here](https://github.com/VITA-Group/TransGAN) for the official Pytorch implementation.


## Dependencies
- Python 3.8
- Tensorfow 2.5


## Usage
### Train
1. Use `--model_name=<name>` to provide the checkpoint directory name. 
```
python train.py --model_name=<name> 
```

#### CIFAR-10
<table>
<tr>
<td style="text-align: center">Epoch 3</td>
<td style="text-align: center">Epoch 50</td> 
<td style="text-align: center">Epoch 500</td>
<td style="text-align: center">Epoch 1975</td> 
</tr>
<trt>
</tr>
<tr>
<td> <img src='images/0003.png'style="width: 400px;"/> </td>
<td> <img src='images/0050.png' style="width: 400px;"/> </td>
<td> <img src='images/0500.png'style="width: 400px;"/> </td>
<td> <img src='images/1975.png' style="width: 400px;"/> </td>
</tr>
</table>

### Hparams setting
Set hyperparameters on the `hparams.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## Implementation notes
- Conditional GAN image generation.


## Licence
MIT