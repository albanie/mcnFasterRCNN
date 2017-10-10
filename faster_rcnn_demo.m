function faster_rcnn_demo(varargin)
%FASTER_RCNN_DEMO Minimalistic demonstration of the Faster R-CNN detector
%   FASTER_RCNN_DEMO an object detection demo with a Faster R-CNN model
%
%   FASTER_RCNN_DEMO(..., 'option', value, ...) accepts the following
%   options:
%
%   `modelPath`:: ''
%    Path to a valid Faster R-CNN matconvnet model. If none is provided, a model
%    will be downloaded.
%
%   `gpus`:: []
%    Device on which to run network 
%
%   `scale`:: 600
%    The minimum size (in pixels) to which the shorter image length will be 
%    resized while preserving aspect ratio
%
%   `maxScale`:: 1000
%    The maximum size (in pixels) to which the longer image length will be 
%    resized while preserving aspect ratio
%
%   `confThresh`:: 0.8
%    The confidence threshold used to determine whether a prediction is 
%    considered a detection
%
%   `nmsThresh`:: 0.3
%    The threshold used to perform non-maximum supression
%
%   `wrapper`:: 'dagnn'
%    The matconvnet wrapper to be used (both dagnn and autonn are supported) 
%
%   `imPath` ::
%    Path to an example image
%
%   `roiVar` :: 'rois'
%    The name of the network variable containing the regions of interests 
%    predicted by the RPN (this name can vary across different pretrained
%    and imported networks, but is typically 'rois' or 'proposals')
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.modelPath = '' ;
  opts.roiVar = 'rois' ;
  opts.scale = 600 ;
  opts.nmsThresh = 0.3 ;
  opts.confThresh = 0.8 ;
  opts.maxScale = 1000 ;
  opts.wrapper = 'dagnn' ;
  opts.imPath = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/misc/000456.jpg') ;
  opts = vl_argparse(opts, varargin) ;

  % The network is trained to prediction occurences
  % of the following classes from the pascal VOC challenge
  classes = {'background', 'aeroplane', 'bicycle', 'bird', ...
     'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
     'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
     'sofa', 'train', 'tvmonitor'} ;

  % Load or download an example faster-rcnn model:
  modelName = 'faster-rcnn-vggvd-pascal.mat' ;
  paths = {opts.modelPath, ...
           modelName, ...
           fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
  ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

  if isempty(ok)
    fprintf('Downloading the Faster R-CNN model ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
    mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
    url = sprintf('%s/~albanie/models/faster/%s', base, modelName) ;
    urlwrite(url, opts.modelPath) ;
  else
    opts.modelPath = paths{ok} ;
  end

  % Load the network with the chosen wrapper
  net = loadModel(opts) ;

  % Load test image
  im = single(imread(opts.imPath)) ;

  % choose variables to track if using dagnn
  if strcmp(opts.wrapper, 'dagnn')
    clsIdx = net.getVarIndex('cls_prob') ;
    bboxIdx = net.getVarIndex('bbox_pred') ;
    roisIdx = net.getVarIndex(opts.roiVar) ;
    [net.vars([clsIdx bboxIdx roisIdx]).precious] = deal(true) ;
  end

  % resize to meet the faster-rcnn size criteria
  imsz = [size(im,1) size(im,2)] ; maxSc = opts.maxScale ; 
  factor = max(opts.scale ./ imsz) ; 
  if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
  newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;

  % resize and subtract mean
  data = imresize(im, factor, 'bilinear') ; 
  data = bsxfun(@minus, data, net.meta.normalization.averageImage) ;

  % set inputs
  sample = {'data', data, 'im_info', imInfo} ;
  net.meta.classes.name = classes ;

  % run network and retrieve results
  switch opts.wrapper
    case 'dagnn' 
      net.eval(sample) ;
      probs = squeeze(net.vars(clsIdx).value) ;
      deltas = squeeze(net.vars(bboxIdx).value) ;
      boxes = net.vars(roisIdx).value(2:end,:)' / imInfo(3) ;
    case 'autonn'
      net.eval(sample, 'test') ;
      probs = squeeze(net.getValue('cls_prob')) ;
      deltas = squeeze(net.getValue('bbox_pred')) ;
      boxes = net.getValue('proposal') ;
      boxes = boxes(2:end,:)' / imInfo(3) ;
  end

  % Visualize results for one class at a time
  for i = 2:numel(classes)
    c = find(strcmp(classes{i}, net.meta.classes.name)) ;
    cprobs = probs(c,:) ;
    cdeltas = deltas(4*(c-1)+(1:4),:)' ;

    cboxes = bbox_transform_inv(boxes, cdeltas);
    cls_dets = [cboxes cprobs'] ;

    keep = bbox_nms(cls_dets, opts.nmsThresh) ;
    cls_dets = cls_dets(keep, :) ;

    sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
    if numel(sel_boxes) == 0, continue ; end

    bbox_draw(im/255,cls_dets(sel_boxes,:));
    title(sprintf('Dets for class ''%s''', classes{i})) ;
    if exist('zs_dispFig', 'file'), zs_dispFig ; end

    fprintf('Detections for category ''%s'':\n', classes{i});
    for j=1:size(sel_boxes,1)
      bbox_id = sel_boxes(j,1);
      fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
              cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
              cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
              cls_dets(bbox_id,end));
    end
  end

% ----------------------------
function net = loadModel(opts)
% ----------------------------
  net = load(opts.modelPath) ; 
  if ~isfield(net, 'forward') % dagnn loader
    net = dagnn.DagNN.loadobj(net) ;
    switch opts.wrapper
      case 'dagnn' 
        net.mode = 'test' ; 
      case 'autonn'
        out = Layer.fromDagNN(net, @extras_autonn_custom_fn) ; 
        net = Net(out{:}) ;
    end
  else % load directly using autonn
    net = Net(net) ;
  end
