import matplotlib
matplotlib.use('Agg')
from os.path import join as pjoin
import os, sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

base = os.path.expanduser('~/coding/libs/py-faster-rcnn2')
caffe_path = pjoin(base, 'caffe-fast-rcnn/python')
zsvision_path = os.path.expanduser('~/coding/src/zsvision/python')
lib_path = pjoin(base, 'lib')
add_path(caffe_path)
add_path(lib_path)
add_path(zsvision_path)

import ipdb
import caffe
from fast_rcnn.config import cfg

pt = 'VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
model = 'faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
prototxt = pjoin(cfg.MODELS_DIR, pt)
caffemodel = pjoin(cfg.DATA_DIR, model)

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
ipdb.set_trace()
