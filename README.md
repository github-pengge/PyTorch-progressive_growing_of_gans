PyTorch implementation of [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://arxiv.org/abs/1710.10196). 

## How to create CelebA-HQ dataset
I borrowed `h5tool.py` from [official code](https://github.com/tkarras/progressive_growing_of_gans). To create CelebA-HQ dataset, we have to download the original CelebA dataset, and the additional deltas files from [here](https://drive.google.com/open?id=0B4qLcYyJmiz0TXY1NG02bzZVRGs). After that, run
```
python2 h5tool.py create_celeba_hq file_name_to_save /path/to/celeba_dataset/ /path/to/celeba_hq_deltas
```
This is what I used on my laptop
```
python2 h5tool.py create_celeba_hq /Users/yuan/Downloads/CelebA-HQ /Users/yuan/Downloads/CelebA/Original\ CelebA/ /Users/yuan/Downloads/CelebA/CelebA-HQ-Deltas
```
I found that MD5 checking were always failed, so I just commented out the MD5 checking part([LN 568](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/h5tool#L568) and [LN 589](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/h5tool#L589))

With default setting, it took 1 day on my server. You can specific `num_threads` and `num_tasks` for accleration.

## Training from scratch
You have to create CelebA-HQ dataset first, please follow the instructions above. 

To obtain the similar results in `samples` directory, see `train_no_tanh.py` or `train.py` scipt for details(with default options). Both should work well. For example, you could run
```
conda create -n pytorch_p36 python=3.6 h5py matplotlib
source activate pytorch_p36
conda install pytorch torchvision -c pytorch
conda install scipy
pip install tensorflow

#0=first gpu, 1=2nd gpu ,2=3rd gpu etc...
python train.py --gpu 0,1,2 --train_kimg 600 --transition_kimg 600 --beta1 0 --beta2 0.99 --gan lsgan --first_resol 4 --target_resol 256 --no_tanh
```

`train_kimg`(`transition_kimg`) means after seeing `train_kimg * 1000`(`transition_kimg * 1000`) real images, switching to fade in(stabilize) phase. Currently only support LSGAN and GAN with `--no_noise` option, since WGAN-GP is unavailable, `--drift` option does not affect the result. `--no_tanh` means do not use `tanh` at generator's output layer.

If you are Python 2 user, You'd better add this to the top of `train.py` since I use print('something...', file=f) to write experiment settings to file.
```
from __future__ import print_function
```


Tensorboard
```
tensorboard --logdir='./logs'
```
## Update history

* **Update(20171213)**: Update `data.py`, now when fading in, real images are weighted combination of current resolution images and 0.5x resolution images. This weighting trick is similar to the one used in Generator's outputs or Discriminator's inputs. This helps stabilize when fading in.

* **Update(20171129)**: Add restoration mode. Basides, after many trying, I failed to combine BEGAN and PG-GAN. It's removed from the repository.

* **Update(20171124)**: Now training with CelebA-HQ dataset. Besides, still failing to introduce progressive growing to BEGAN, even with many modifications.

* **Update(20171121)**: Introduced progressive growing to [BEGAN](https://arxiv.org/abs/1703.10717), see `train_began.py` script. However, experiments showed that it did not work at this moment. Finding bugs and tuning network structure...

* **Update(20171119)**: Unstable came from `resize_activation` function, after replacing `repeat` by `torch.nn.functional.upsample`, problem solved. **And now I believe that both `train.py` and `train_no_tanh` should be stable**. Restored from 128x128 stabilize, and continued training, currently at 256x256, phase = fade in, temporary results(first 2 columns on the left were generated, and the other 2 columns were taken from dataset):

<p align="center">
  <img src="/samples/256x256-fade_in-092000.png">
</p>
<p align="center">
  <img src="/samples/256x256-fade_in-092500.png">
</p>

* **Update(20171118)**: Making mistake in `resize activation` function(`repeat` is not a right in this function), though it's wrong, it's still effective when resolution<256, but collapsed at resolution>=256. Changing it now, scripts will be updated tomorrow. Sorry for this mistake.

* **Update(20171117)**: 128x128 fade in results(first 2 columns on the left were generated, and the other 2 columns were taken from dataset):

<p align="center">
  <img src="/samples/128x128-fade_in-134500.png">
</p>
<p align="center">
  <img src="/samples/128x128-fade_in-135000.png">
</p>

* **Update(20171116)**: Adding noise only to RGB images might still collapse. Switching to the same trick as the paper suggested. Besides, the paper used `linear` as activation of G's output layer, which is reasonable, as I observed in the experiments. Temporary results: 64x64, phase=fade in, the left 4 columns are Generated, and the right 4 columns are from real samples(when fading in, instability might occur, for example, the following results is not so promising, however, as the training goes, it gets better), higher resolution will be available soon.

<p align="center">
  <img src="/samples/64x64-fade_in-060000.png">
  <img src="/samples/64x64-fade_in-072500.png">
</p>

* **Update(20171115)**: Mode collapse happened when fading in, debugging... => It turns out that unstable seems to be normal when fading in, after some more iterations, it gets better. Now I'm not using the same noise adding trick as the paper suggested, however, it had been implemented, I will test it and plug it into the network.

* **Update(20171114)**: First version, seems that the generator tends to generate white image. Debugging now. => Fixed some bugs. Now seems normal, training... => There are some unknown problems when fading in, debugging...

* **Update(20171113)**: Generator and Discriminator: ok, simple test passed.

* **Update(20171112)**: It's now under reimplementation.

* **Update(20171111)**: It's still under implementation. I did not care design the structure, and now I had to reimplement(phase='fade in' is hard to implement under current structure). I also fixed some bugs, since reimplementation is needed, I do not plan to pull requests at this moment.

# Reference implementation
* https://github.com/github-pengge/PyTorch-progressive_growing_of_gans


