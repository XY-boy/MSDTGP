# MSDTGP (IEEE TGRS 2022)
### 📖[**Paper**](https://ieeexplore.ieee.org/abstract/document/9530280) | 🖼️[**PDF**](/img/MSDTGP.pdf) | 🎁[**Dataset**](https://zenodo.org/record/6969604)

PyTorch codes for "[Satellite Video Super-resolution via Multi-Scale Deformable Convolution Alignment and Temporal Grouping Projection](https://ieeexplore.ieee.org/abstract/document/9530280)", **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, 2022.

Authors: [Yi Xiao](https://xy-boy.github.io/), Xin Su*, [Qiangqiang Yuan](http://qqyuan.users.sgg.whu.edu.cn/), Denghong Liu, [Huanfeng Shen](https://scholar.google.com.hk/citations?user=ore_9NIAAAAJ&hl), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
Wuhan University
### Abstract
> As a new earth observation tool, satellite video has been widely used in remote-sensing field for dynamic analysis. Video super-resolution (VSR) technique has thus attracted increasing attention due to its improvement to spatial resolution of satellite video. However, the difficulty of remote-sensing image alignment and the low efficiency of spatial–temporal information fusion make poor generalization of the conventional VSR methods applied to satellite videos. In this article, a novel fusion strategy of temporal grouping projection and an accurate alignment module are proposed for satellite VSR. First, we propose a deformable convolution alignment module with a multiscale residual block to alleviate the alignment difficulties caused by scarce motion and various scales of moving objects in remote-sensing images. Second, a temporal grouping projection fusion strategy is proposed, which can reduce the complexity of projection and make the spatial features of reference frames play a continuous guiding role in spatial–temporal information fusion. Finally, a temporal attention module is designed to adaptively learn the different contributions of temporal information extracted from each group. Extensive experiments on Jilin-1 satellite video demonstrate that our method is superior to current state-of-the-art VSR methods.
### Network  
 ![image](/img/network.png)
## 🧩Install
```
git clone https://github.com/XY-boy/MSDTGP.git
```
## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 
 ## Dataset Preparation
 Please download our dataset in 
 * Baidu Netdisk [Jilin-189](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) Code:31ct
 * Zenodo: <a href="https://doi.org/10.5281/zenodo.6969604"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6969604.svg" alt="DOI"></a>
 
You can also train your dataset following the directory structure below!
 
### Data directory structure
trainset--  
&emsp;|&ensp;train--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 

testset--  
&emsp;|&ensp;eval--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 
 ## Training
```
python main.py
```

## Test
```
python eval.py
```
### Quantitative results
 ![image](/img/res1png.png)
 
 ### Qualitative results
 ![image](/img/res2.png)
 #### More details can be found in our paper!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work useful in your research, we would appreciate your citation. 😊😊
```
@ARTICLE{xiao2022msdtgp,  
author={Xiao, Yi and Su, Xin and Yuan, Qiangqiang and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},  
journal={IEEE Transactions on Geoscience and Remote Sensing},  
title={Satellite Video Super-Resolution via Multiscale Deformable Convolution Alignment and Temporal Grouping Projection},   
year={2022},  
volume={60},  
number={},  
pages={1-19},  
doi={10.1109/TGRS.2021.3107352}
}
```

## Acknowledgement
Our work is built upon [RBPN](https://github.com/alterzero/RBPN-PyTorch) and [TDAN](https://github.com/YapengTian/TDAN-VSR-CVPR-2020).  
Thanks to the author for the source code !
