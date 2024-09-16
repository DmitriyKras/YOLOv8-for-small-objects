# YOLOv8-for-small-objects
This repository contains implementation for Dmitrii I. Krasnov, Sergey N. Yarishev, Victoria A. Ryzhova,  Todor S. Djamiykov paper *"Improved YOLOv8 Network for Small Objects Detection"* (link to publication is coming soon). Repo is a modification of original [Ultralytics](https://github.com/ultralytics/ultralytics) so `ultralytics` folder remains unchanged (almost).


## Abstract
This project is an implementation of the experimantal adaptation pipeline for object detection and semantic segmentation models. 
This pipeline specifically designed to improve model's architecture in order to adapt it to small objects.

## Steps

### P2 detection head

P2 detection head and assosiated tensors pathes (W / 4, H / 4 of image resolution) added to the original network in order to increase number of small-sized bboxes detected. This step increases overall number of bounding boxes and slows down inference and post-process. However, resulting speed is enough for most of the practical applications (see benchmarks).

### P5 detection head

Simply remove it! P2 detection head and assosiated tensors responsible for large objects detection which is redundant for current task. Evaluation results shows, that this step is compensated with P2 detection head so there is no losses. Moreover this step helps to reduce overall number of model's paremeters by 46.8%.

### Bi-directional feature fusion

Concat layer in original YOLOv8 model was replaced with Bi-directrional concat. This block use feature maps with both higher and lower resolutions to fuse them with feature map of current resolution (see figure). This step allows simultaneous use of the deep and shallow features.

![Bi-directional feature fusion](/assets/bi_directional_fusion.png)

### CBAM attention module

A main building block of YOLOv8 C2f was enhanced with CBAM attention module. This module is a sequence of the channel attention block and spatial attention block. Attention module is necessary for supressing less informative channels and areas on feature maps.

![CBAM](/assets/cbam.png)

