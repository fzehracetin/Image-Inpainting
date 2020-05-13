# Image Inpainting

In this project I will be using 3 inpainting methods on MIT Places Dataset. 2 inpainting methods are based on neural networks. On the other hand, 1 method based on statistics.

#### Neural Methods
1. [Generative Inpainting (2018)](https://github.com/JiahuiYu/generative_inpainting)

For this approach I worked with the offical code and design. So I won't add the same codes to this repository. You can check code files from their Github and follow the steps for training. My test results and test codes will be under [this directory.](https://github.com/fzehracetin/Image-Inpainting/tree/master/Generative%20Inpainting)

2. [Partial Convolutions (2018)](https://github.com/NVIDIA/partialconv)

For this research, Nvidia did not share all the code files for inpainting. So I worked with unofficial repository, [MathiasGruber's PConv-Keras repo.](https://github.com/MathiasGruber/PConv-Keras). I made changes on Step 4 (Training part) and I rewrote Test part. If you work with 256, 256 images like me, you should change the img_rows and img_cols variables in the libs/pconv_model.py line 21. The code files that I changed and test results will be under [this directory.](https://github.com/fzehracetin/Image-Inpainting/tree/master/Partial%20Convolutions)

#### Statistical Method
3. [Statistics of Patch Offsets (2012)](https://github.com/Pranshu258/Image_Completion)

This method is easy to work, because there is no training process. You have to use Python2. In Windows I encountered so many erors about PyMaxflow in the installation process, it was related to Visual C++. So, I used Ubuntu, it was much smooth. I edited code little bit, to work with all images under a directory instead of a single image. Code file is under [this directory.](https://github.com/fzehracetin/Image-Inpainting/tree/master/Statistics%20of%20Patch%20Offsets) You must make changes in config.py. Code is very interwined with this file. So I didn't break this connection between them. 

If you have any question about implementation of this methods don't hesitate to ask me. Good luck! :stars:
