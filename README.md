# TransGAN
Implementation of the paper:

> Yifan Jiang, Shiyu Chang and Zhangyang Wang. [TransGAN: Two Pure Transformers Can Make One
Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074). 

![Architecture](./images/architecture.png)

This repository implements Small-TransGAN and is meant to be educational. See _Implementation notes_.

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

### Hparams setting
Set hyperparameters on the `hparams.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## Examples
### CIFAR-10
Small-TransGAN training progress

![](images/small_transgan.gif "Small-TransGAN on CIFAR-10")


## Implementation notes
Code:
- This model depends on other files that may be licensed under different open source licenses.
- TransGAN uses Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu and Song Han. [Differentiable Augmentation](https://arxiv.org/abs/2006.10738). Under BSD 2-Clause "Simplified" License.
- Small-TransGAN models are instances of the original TransGAN architecture with a smaller number of layers and lower-dimensional embeddings.

To train Small-TransGAN:
- Use single layer per resolution Generator.
- Increase the MLP hidden dimension of the Generator last block.
- Use orthogonal initializer and 4 heads in both Generator and Discriminator.
- Employ WGAN-GP loss.
- Adam with β1 = 0.0 and β2 = 0.99.
- Set noise vector dimension to 64.


## Licence
MIT