function net = faster_rcnn_deploy(srcPath, destPath, varargin)
% FASTER_RCNN_DEPLOY deploys a FASTER_RCNN model for evaluation
%   NET = FASTER_RCNN_DEPLOY(SRCPATH, DESTPATH) configures
%   a Faster-RCNN model to perform evaluation.  This process involves
%   removing the loss layers used during training and adding 
%   a combination of a transpose softmax with a detection
%   layer to compute network predictions.
%
%   FASTER_RCNN_DEPLOY(..'name', value) accepts the following 
%   options:
%
%   `toDagNN` :: true
%    If true, the network is stored as a DagNN object, rather 
%    than as an AutoNN object.
%
%   `precomputedBboxNormalization` :: true
%    To ease the optimization, during learning bounding boxes are normalized
%    according to a precomputed set of fixed means and standard deviations.
%    If this option is true (indicating the normalization was used during 
%    training), during deployment this process is undone and the regressors 
%    are recalibrated to operate on proposals.
%
%   `normalizeMeans` :: [0, 0, 0, 0]
%    The means used to normalize the bounding boxes.
%
%   `normalizeStdDevs` :: [0.1, 0.1, 0.2, 0.2]
%    The standard deviations used to normalize the bounding boxes.
%
%   `numClasses` :: 21
%    The number of classes that the network was trained to predict (used
%    as part of the bbox regressor normalisation.
%
%   `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%    The path to the directory containing the coco data
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.toDagNN = true ;
  opts.numClasses = 21 ;
  opts.normalizeMeans = [0, 0, 0, 0] ;
  opts.normalizeStdDevs = [0.1, 0.1, 0.2, 0.2] ;
  opts.precomputedBboxNormalization = true ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  tmp = load(srcPath) ; 
  out = Layer.fromCompiledNet(tmp.net) ; frcnn = out{2} ;

  % fix names from old config
  map = {{'proposals', 'proposal'}, {'imInfo', 'im_info'}} ;
  for ii = 1:numel(map)
    pair = map{ii} ; old = pair{1} ; new = pair{2} ;
    if ~isempty(frcnn.find(old)), frcnn.find(old, 1).name = new ; end
  end

  % modify network to use RPN for predictions at test time
  frcnn.find('proposal', 1).inputs{7} = 300 ; % num proposals
  frcnn.find('proposal', 1).inputs{9} = 6000 ; % pre-NMS top N
  frcnn.find('roi_pool5',1).inputs{2} = frcnn.find('proposal',1) ;

  % set outputs
  bbox_pred = frcnn.find('bbox_pred', 1) ;
  cls_score = frcnn.find('cls_score', 1) ;

  % unnormalize bbox predictors
  if ~opts.precomputedBboxNormalization, error('unsupported') ; end
  m = repmat(opts.normalizeMeans', opts.numClasses, 1) ;
  std = repmat(opts.normalizeStdDevs', opts.numClasses, 1) ;
  f = bbox_pred.inputs{2}.value ; b = bbox_pred.inputs{3}.value ;

  % biases have shape (4*numClasses)x1, so operate along first dim
  bbox_pred.inputs{3}.value = bsxfun(@plus, bsxfun(@times, b, std), m) ;

  % filters have shape (1,1,C,4*numClasses), so operate along last dim 
  m_ = permute(m, [4 3 2 1]) ; std_ = permute(std, [4 3 2 1]) ; 
  bbox_pred.inputs{2}.value = bsxfun(@plus, bsxfun(@times, f, std_), m_) ;

  % normalze to probabilities
  largs = {'name', 'cls_prob', 'numInputDer', 0} ;
  cls_prob = Layer.create(@vl_nnsoftmax, {cls_score}, largs{:}) ;
  net = Net(cls_prob, bbox_pred) ;

  outDir = fileparts(destPath) ;
  if ~exist(outDir, 'dir'), mkdir(outDir) ; end

  net.meta.backgroundClass = 1 ; 

  % add standard imagenet average if not present
  if ~isfield(net.meta, 'normalization'), net.meta.normalization = struct() ; end
  if ~isfield(net.meta.normalization, 'averageImage')
    rgb = [122.771, 115.9465, 102.9801] ; 
    net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;
  end

  if opts.toDagNN
    customDagObj = faster_rcnn_dagnn_custom() ;
    net = toDagNN(net, customDagObj) ;
  end
  net = net.saveobj() ; 
  save(destPath, '-struct', 'net') ;
