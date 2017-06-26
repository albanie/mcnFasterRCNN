Faster R-CNN
---

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

The model currently only operates in test mode, using the pre-trained models 
released with the caffe code which have been imported into matconvnet and 
can be downloaded [here](http://www.robots.ox.ac.uk/~albanie/models.html#faster-rcnn-models).

Training code will be added when the original experiments have been reproduced.
