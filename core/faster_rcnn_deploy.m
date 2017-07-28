function net = faster_rcnn_deploy(srcPath, destPath, varargin)
% FASTER_RCNN_DEPLOY deploys a FASTER_RCNN model for evaluation
%   NET = FASTER_RCNN_DEPLOY(SRCPATH, DESTPATH) configures
%   a Faster-RCNN model to perform evaluation.  THis process involves
%   removing the loss layers used during training and adding 
%   a combination of a transpose softmax with a detection
%   layer to compute network predictions
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.numClasses = 21 ;
  opts.precomputedBboxNormalization = 1 ;
  opts.normalizeMeans = [0, 0, 0, 0] ;
  opts.normalizeStdDevs = [0.1, 0.1, 0.2, 0.2] ;
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

  if ~opts.precomputedBboxNormalization, error('unsupported') ; end

  % unnormalize bbox predictors
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
  net = net.saveobj() ; 
  save(destPath, '-struct', 'net') ;
