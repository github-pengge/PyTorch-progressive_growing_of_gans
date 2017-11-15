# Progressive growing of GANs
PyTorch implementation of [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://arxiv.org/abs/1710.10196). 

**Update(20171115)**: Mode collapse happened when fading in, debugging...

**Update(20171114)**: First version, seems that the generator tends to generate white image. Debugging now. => Fixed some bugs. Now seems normal, training... => There are some unknown problems when fading in, debugging...

**Update(20171113)**: Generator and Discriminator: ok, simple test passed.

**Update(20171112)**: It's now under reimplementation.

**Update(20171111)**: It's still under implementation. I did not care design the structure, and now I had to reimplement(phase='fade in' is hard to implement under current structure). I also fixed some bugs, since reimplementation is needed, I do not plan to pull requests at this moment.

# Official implementation
Official implementation using lasagne can ben found at [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans).

