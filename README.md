# Satellite Video Super-resolution via Multi-Scale Deformable Convolution Alignment and Temporal Grouping Projection (TGRS)
## Introuction
This is the official implementation of our paper [Satellite Video Super-resolution via Multi-Scale Deformable Convolution Alignment and Temporal Grouping Projection](https://ieeexplore.ieee.org/document/9530280) published on IEEE Transactions on Geoscience and Remote Sensing (**TGRS**).  

### The network structure  
 ![image](/img/network.png)
 
 ### Quantitive results
 ![image](/img/res1png.png)
 
 ### Qualitive results
 ![image](/img/res2.png)
 #### More details can be found in our paper!
 ## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 
 ## Dataset Preparation
 Please download our dataset [Jilin-189](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) Code:31ct  
 You can also train your dataset following the directory sturture below!
 
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

## Citation
If you find our work helpful, please cite:  
```
@ARTICLE{9530280,  
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



 


