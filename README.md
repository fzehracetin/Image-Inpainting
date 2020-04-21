# Image Inpainting

In this project I will be using 4 inpainting methods on MIT Places Dataset. 3 inpainting methods are based on neural networks. On the other hand, 1 method based on statistics.

#### Neural Methods
1. [Generative Inpainting (2018)](https://github.com/JiahuiYu/generative_inpainting)

For this approach I worked with the offical code and design. So I won't add the same codes to this repository. You can check code files from their Github and follow the steps for training. My test results and test codes will be under [this directory.](https://github.com/fzehracetin/Image-Inpainting/tree/master/Generative%20Inpainting)

2. [Partial Convolutions (2018)](https://github.com/NVIDIA/partialconv)

For this research, Nvidia did not share all the code files for inpainting. So I worked with unofficial repository, [MathiasGruber's PConv-Keras repo.](https://github.com/MathiasGruber/PConv-Keras). I made changes on Step 4 (Training part) and I rewrote Test part. If you work with 256, 256 images like me, you should change the img_rows and img_cols variables in the libs/pconv_model.py line 21. The code files that I changed and test results will be under [this directory.](https://github.com/fzehracetin/Image-Inpainting/tree/master/Partial%20Convolutions)


3. [High-Resolution Neural Inpainting (2017)](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)
#### Statistical Method
4. [Statistics of Patch Offsets (2012)](https://github.com/Pranshu258/Image_Completion)
