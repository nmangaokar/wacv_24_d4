# 	[IEEE/CVF WACV 2024] D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles

This is an implementation of the D4 framework described in the IEEE/CVF WACV 2024 paper:
[D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles](https://openaccess.thecvf.com/content/WACV2024/papers/Hooda_D4_Detection_of_Adversarial_Diffusion_Deepfakes_Using_Disjoint_Ensembles_WACV_2024_paper.pdf).

## 1. Environment

I'd recommend having something that resembles the environment described below to make things easy:

- **OS:** Ubuntu 20.04.6 LTS
- **CPU:** Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz 
- **GPU:** NVIDIA GeForce RTX 2080 Ti Rev. A w/ 11 GB VRAM
- **Conda:** 4.9.2

Set up your environment using the conda environment file `environment.yml` as follows:

```conda env create -f environment.yml```

After you're all set up, go ahead and activate the `d4` environment to run things:

```conda activate d4```

## 2. Code

### 2.1. Usage
You'll want to run things by invoking `main.py` with a config file as follows:

`python main.py --config [path to config file] --start_idx [start index] --end_idx [end index]`

Arguments are hopefully self-explanatory:

```
--config (str): Path to a config file that specifies the task (training, etc) and the parameters for the task 
--start_idx (int): Index of the first image in the dataset to be attacked 
--end_idx (int): Index of the last image in the dataset to be attacked 
```

### 2.2. Tasks

Each config file describes a task setting. Ex - `configs/d4.py`. There are 4 types of tasks:
1. `train`: model training, for example look at `configs/at.py`
2. `saliency`: compute the saliency values and frequency splits
3. `nat_eval`: evaluate the accuracy on natural images
4. `adv_eval`: evaluate performance against attacks, for example look at `configs/d4.py`

# 3. Responsible Use, License, and Citation
This repository contains attack code. In general, please be responsible while executing this code and do not
use it on classification services for which you do not have permission. It is meant to be used for research purposes only. 
This code is licensed under the MIT license:

> MIT License
> 
> Copyright (c) 2023 Authors of "D4: Detection of Adversarial..."
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.


If you use this code in your research, please cite the following paper:

```
@InProceedings{Hooda_2024_WACV,
    author    = {Hooda, Ashish and Mangaokar, Neal and Feng, Ryan and Fawaz, Kassem and Jha, Somesh and Prakash, Atul},
    title     = {D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {3812-3822}
}
