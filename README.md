Faster R-CNN
---

**NOTICE: This repo is no longer maintained.  For an easy-to-use object detector that is actively maintained, I recommend considering the PyTorch Mask R-CNN [implementation](https://github.com/facebookresearch/maskrcnn-benchmark).**

This directory contains code to evaluate the Faster R-CNN object detector 
described in the paper:

```
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,
Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun,
Advances in Neural Information Processing Systems (NIPS), 2015
```

This code is based on the `py-caffe` implementation 
[made available](https://github.com/rbgirshick/py-faster-rcnn) by 
[Ross Girshick](http://www.rossgirshick.info/) and the MatConvNet 
[Fast R-CNN implementation](https://github.com/vlfeat/matconvnet/tree/master/examples/fast_rcnn) by 
[Hakan Bilen](http://www.robots.ox.ac.uk/~hbilen).

The pre-trained models released with the caffe code which have been imported 
into matconvnet and can be downloaded
[here](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#faster-rcnn-models),
together with models trained directly with this code.  Alternatively, you can 
train your own detector.

### Demo

Running the `faster_rcnn_demo.m` script will download a model trained on pascal voc 2007 data and run it on a sample image to produce the figure below:

<img src="misc/pascal-demo-fig.jpg" width="600" />

### Functionality

There are scripts to evaluate models on the `pascal voc` and `ms coco` datasets (the scores produced by the pretrained models are listed on the [model page](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#faster-rcnn-models)).  Training code is also provided to reproudce the `pascal voc` experiments described in the paper.  In addition, there is the option to train with "SSD-style" zoom augmentation to the improve performance of the model beyond the original baseline.


### Dependencies

Faster-RCNN requires the following modules:

* [autonn](https://github.com/vlfeat/autonn) - automatic differentiation
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) - extra MatConvNet layers
* [mcnDatasets](https://github.com/albanie/mcnDatasets) - dataset handlers
* [GPU NMS](https://github.com/albanie/mcnNMS) - a CUDA-based implementation of non-maximum suppression

If you wish to provide more extensive data augmentation, the following module is required (it can also be installed with `vl_contrib`):

* [mcnSSD](https://github.com/albanie/mcnSSD) - SSD detector implementation (provides data augmentation sampler)

The effect of the CUDA NMS module is discussed below.

### Installation

The easiest way to use this module is to install it with the `vl_contrib`
package manager. `mcnFasterRCNN` can be installed with the following commands
rom the root directory of your MatConvNet installation:

```
vl_contrib('install', 'mcnFasterRCNN') ;
vl_contrib('setup', 'mcnFasterRCNN') ;
vl_contrib('test', 'mcnFasterRCNN') ; % optional
```  


### Accuracy

A comparison of the mean AP of the trained detectors is given [here](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#faster-rcnn-models).   The following numbers were obtained from a single run of both implementations (there may be some variance if repeated):

| training code | voc 07 test mAP |  
|---------------|-----------------|
| py-caffe      |     69.7 mAP    |  
| matconvnet    |     69.3 mAP    |  
| matconvnet+   |     70.2 mAP    |  

Two matconvnet models are presented above. The first, labeled `matconvnet`, is trained without "SSD-style" data augmentation (discussed in more detail below) and uses only the flip augmentation used in the py-caffe implementation.  The second, `matconvnet+`, uses the augmentation pipeline.  This can be switched on to improve beyond the original baseline, but requires a longer training schedule.


### Efficiency

The Faster R-CNN pipeline makes heavy use of non-maximum suppression during training and inference. As a result, the runtime of the detector is significantly affected by the efficiency of the NMS function.  A GPU version of non-maximum suppression can be found [here](https://github.com/albanie/mcnNMS), which can be compiled and added to your MATLAB path.  Approximate benchmarks (they do not currently include the decoding of the raw predictions) of the code are given below on a Tesla M40 with a single item batch size:


| mode      | NMS (CPU) | NMS (GPU) |  
|-----------|-----------|-----------|  
| training  | 1.1 Hz    | 3.1 Hz    |  
| inference | 6.7 Hz    | 8.6 Hz    |  


 Running the detector with on multiple GPUs produces a significant speed boost during inference, but currently only a minimal improvement during training (this may be addressed in future). 

**Multi-GPU code performance:**

| mode      | Single GPU | 2 GPUs   |
|-----------|-----------|-----------|
| training  | 3.1 Hz    | 3.7 Hz    |
| inference | 8.6 Hz    | 15 Hz     |


### Data Augmentation

The [SSD detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2) introduced, among other things, a form of aggressive data augmentation designed to improve the quality of the trained detector.  The "zoom augmentation" technique introduced in the paper is implemented in this code and can be applied directly to the Faster R-CNN detector.  The implmentation is fairly efficient, since all image resampling is performed on the GPU.  However it is not without cost, and reduces training speed by approximately 5%.  An example of the effect of zoom augmentation is shown below:

![zoom-aug](misc/zoom-aug.png)

The full details can be found in the SSD paper linked above.   Essentially the original image and corresponding bounding boxes are shrunk and randomly placed in a mean-pixel value canvas.  This is combined with patch augmentation and colour distortion to further help generalisation. 



