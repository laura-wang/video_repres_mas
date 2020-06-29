#  Self-Supervised Spatio-Temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics
Tensorflow implementation of our CVPR 2019 paper [Self-Supervised Spatio-Temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics.](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Self-Supervised_Spatio-Temporal_Representation_Learning_for_Videos_by_Predicting_Motion_and_CVPR_2019_paper.html)

## Update

Pytorch implemetation https://github.com/laura-wang/video_repres_sts


## Overview
We realease partial of our training code on UCF101 dataset. It contains the self-supervised learning based on motion statistics (see more details in our paper).    
The entire training protocol (both motion statistics and appearance statistics) is implemented in the pytorch version: https://github.com/laura-wang/video_repres_sts.

## Requirements
1. tensorflow >= 1.9.0
2. Python 3
3. cv2
4. scipy

## Data preparation

You can download the original UCF101 dataset from the [official website](https://www.crcv.ucf.edu/data/UCF101.php). And then extarct RGB images from videos and finally extract optical flow data using TVL1 method. **But I recommend you to direclty download the pre-processed RGB and optical flow data of UCF101 provided by [feichtenhofer](https://github.com/feichtenhofer/twostreamfusion).** 

## Train
Here we provide the first version of our training code with "placeholder" as data reading pipeline, so you don't need to write RGB/Optical flow data into tfrecord format. We also rewrite the training code using Dataset API, but currently we think the placeholder version is enough for you to get to understand motion statsitics. 

Before `python train.py`, remember to set right dataset directory in the list file, and then you can play with the motion statistics!

## Citation

If you find this repository useful in your research, please consider citing:

```
@inproceedings{wang2019self,
  title={Self-Supervised Spatio-Temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics},
  author={Wang, Jiangliu and Jiao, Jianbo and Bao, Linchao and He, Shengfeng and Liu, Yunhui and Liu, Wei},
  booktitle={CVPR},
  pages={4006--4015},
  year={2019}
}
```


