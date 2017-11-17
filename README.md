# Progressive growing of GANs
PyTorch implementation of [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://arxiv.org/abs/1710.10196). 

* **Update(20171117)**: 128x128 fade in results(first 2 columns on the left were generated, and the other 2 columns were taken from dataset):

<p align="center">
  <img src="/samples/128x128-fade_in-134500.png">
</p>
<p align="center">
  <img src="/samples/128x128-fade_in-135000.png">
</p>

**Setup for the results above**: see `train_no_tanh.py` scipt for details(with default options).

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

# Official implementation
Official implementation using lasagne can ben found at [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans).

