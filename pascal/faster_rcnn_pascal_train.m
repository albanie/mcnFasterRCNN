function faster_rcnn_pascal_train(varargin)
%FASTER_RCNN_PASCAL_TRAIN Train a Faster R-CNN detector on pascal
%   FASTER_RCNN_PASCAL_TRAIN performs a full training run of a Faster R-CNN
%   detector on the Pascal VOC dataset. A number of options and settings are
%   provided for training.  The defaults should reproduce the experiment
%   described in the original Faster R-CNN paper (linked in README.md).
%
%   FASTER_RCNN_PASCAL_TRAIN(..'name', value) accepts the following
%   options:
%
%   `pruneCheckpoints` :: true
%    Determines whether intermediate training files should be cleared to save
%    space after the training run has completed.
%
%   `nms` :: 'gpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `confirmConfig` :: true
%    Ask the user to confirm the experimental settings before running the
%    experiment
%
%   `checkAgainstProto` :: false
%    If available, will check all learning parameters against the prototxt
%    of the original models released by Ross Girshick.  This can be useful
%    during development, but should not be necessary for the standard
%    architectures which have been verified experimentally.

% ----------------------------------------------------------------------
%   `train` :: struct(...)
%    A structure of options for training, with the following fields:
%
%      `gpus` :: 1
%       If provided, the gpu ids to be used for processing.
%
%      `batchSize` :: 1
%       Number of images per batch during training.
%
%      `continue` :: true
%       Resume training from previous checkpoint.
%
% ----------------------------------------------------------------------
%   `dataOpts` :: struct(...)
%    A structure of options for the data, with the following fields:
%
%      `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%       The path to the directory containing the Pascal VOC data data
%
%      `useValForTraining` :: true
%       Whether the validation set (as defined in the original challenge)
%       should be included in the training set.
%
%      `zoomScale` :: 2
%       Zoom magnitude used by SSD-style data augmentation
%
%      `flipAugmentation` :: true
%       Whether flipped images should be used in the training procedure.
%
%      `zoomAugmentation` :: false
%       Use "zoom" augmentation to improve performance (but longer training)
%
%      `patchAugmentation` :: false
%       Use SSD-style "patch" augmentation to improve performance
%
%      `distortAugmentation` :: false
%       Use SSD-style "distortion" augmentation to improve performance
%
% ----------------------------------------------------------------------
%   `modelOpts` :: struct(...)
%    A structure of options for the model, with the following fields:
%
%      `architecture` :: 'vgg16'
%       The trunk architecture used to initialise the model.
%
%      `classAgnosticReg` :: false
%       Whether to train a class agnostic bounding box regressor (in the style
%       used by R-FCN, or per-class bounding box regressors (in the style of
%       Faster R-CNN).
%
%      `roiBatchSize` :: 128
%       The number or "regions-of-interest" sampled per batch during the
%       training of the RPN.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.debug = 0 ;
  opts.nms = 'gpu' ; % set to CPU if mcnNMS module is not installed
  opts.continue = 1 ;
  opts.confirmConfig = false ;
  opts.pruneCheckpoints = true ;
  opts.checkAgainstProto = false ;

  % configure training options
  opts.train.gpus = 2 ;
  opts.train.derOutputs = 1 ; % give each loss the same weight
  opts.train.batchSize = numel(opts.train.gpus) ;
  opts.train.numSubBatches = numel(opts.train.gpus) ;
  opts.train.continue = opts.continue ;
  opts.train.parameterServer.method = 'mmap' ;
  opts.train.stats = {'rpn_loss_cls', 'rpn_loss_bbox','loss_cls', 'loss_bbox', ...
                      'multitask_loss', 'cls_error'} ; % train end to end

  % configure dataset options
  opts.dataOpts.name = 'pascal' ;
  opts.dataOpts.trainData = '07' ;
  opts.dataOpts.testData = '07' ;
  opts.dataOpts.flipAugmentation = true ;
  opts.dataOpts.zoomAugmentation = false ;
  opts.dataOpts.patchAugmentation = false ;
  opts.dataOpts.distortAugmentation = false ;
  opts.dataOpts.useValForTraining = true ;
  opts.dataOpts.getImdb = @getCombinedPascalImdb ; %@getPascalImdb ;
  opts.dataOpts.prepareImdb = @prepareImdb ;
  opts.dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;
  opts.dataOpts.zoomScale = 2 ;

  % configure model options
  opts.modelOpts.type = 'faster-rcnn' ;
  opts.modelOpts.nms = opts.nms ;
  opts.modelOpts.locWeight = 1 ;
  opts.modelOpts.numClasses = 21 ;
  opts.modelOpts.featStride = 16 ;
  opts.modelOpts.ratios = [0.5, 1, 2] ;
  opts.modelOpts.scales = [8, 16, 32] ;
  opts.modelOpts.clipPriors = false ;
  opts.modelOpts.net_init = @faster_rcnn_init ;
  opts.modelOpts.deploy_func = @faster_rcnn_deploy ;
  opts.modelOpts.batchSize = opts.train.batchSize ;
  opts.modelOpts.get_batch = @faster_rcnn_train_get_batch ;
  opts.modelOpts.architecture = 'vgg16' ;
  opts.modelOpts.batchNormalization = false ;
  opts.modelOpts.batchRenormalization = false ;
  opts.modelOpts.instanceNormalization = false ;
  opts.modelOpts.CudnnWorkspaceLimit = 512*1024*1204 ; % 1GB/512MB
  opts.modelOpts.freezeBnorm = 0 ;
  opts.modelOpts.roiBatchSize = 128 ;
  opts.modelOpts.initMethod = 'gaussian' ;
  opts.modelOpts.classAgnosticReg = false ;
  opts = vl_argparse(opts, varargin) ;

  protoName = sprintf('%s_train.prototxt', opts.modelOpts.architecture) ;
  protoDir = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/misc') ;
  opts.modelOpts.protoPath = fullfile(protoDir, protoName) ;

  % sanity checks
  msg = 'cannot use both batch and instance normalization' ;
  assert(~(opts.modelOpts.instanceNormalization ...
                   && opts.modelOpts.batchNormalization), msg) ;
  % don't merge if using Instance Norm
  opts.modelOpts.mergeBnorm = ~opts.modelOpts.instanceNormalization ;

  % Set learning rates
  steadyLR = 0.001 ;
  gentleLR = 0.0001 ;
  vGentleLR = 0.00001 ;

  if ~opts.dataOpts.zoomAugmentation
    % this should correspond (approximately) to the 70,000 iterations
    % used in the original model (when zoom aug is not used)
    numSteadyEpochs = 10 ;
    numGentleEpochs = 4 ;
    numVeryGentleEpochs = 0 ;
  else
    % "SSD-style" data augmentation uses a longer training schedule
    numSteadyEpochs = 20 ;
    numGentleEpochs = 5 ;
    numVeryGentleEpochs = 5 ;
  end

  steady = steadyLR * ones(1, numSteadyEpochs) ;
  gentle = gentleLR * ones(1, numGentleEpochs) ;
  veryGentle = vGentleLR * ones(1, numVeryGentleEpochs) ;
  opts.train.learningRate = [steady gentle veryGentle] ;
  opts.train.numEpochs = numel(opts.train.learningRate) ;

  % configure batch opts
  batchOpts.scale = 600 ;
  batchOpts.maxScale = 800 ; % cheaper and avoids memory crashes
  batchOpts.patchOpts.use = opts.dataOpts.patchAugmentation ;
  batchOpts.patchOpts.numTrials = 50 ;
  batchOpts.patchOpts.minPatchScale = 0.3 ;
  batchOpts.patchOpts.maxPatchScale = 1 ;
  batchOpts.patchOpts.minAspect = 0.5 ;
  batchOpts.patchOpts.maxAspect = 2 ;
  batchOpts.patchOpts.clipTargets = true ;
  batchOpts.flipOpts.use = opts.dataOpts.flipAugmentation ;
  batchOpts.flipOpts.prob = 0.5 ;
  batchOpts.zoomOpts.use = opts.dataOpts.zoomAugmentation ;
  batchOpts.zoomOpts.prob = 0.5 ;
  batchOpts.zoomOpts.minScale = 1 ;
  batchOpts.zoomOpts.maxScale = opts.dataOpts.zoomScale ;
  batchOpts.distortOpts.use = opts.dataOpts.distortAugmentation ;
  batchOpts.distortOpts.brightnessProb = 0.5 ;
  batchOpts.distortOpts.contrastProb = 0.5 ;
  batchOpts.distortOpts.saturationProb = 0.5 ;
  batchOpts.distortOpts.hueProb = 0.5 ;
  batchOpts.distortOpts.brightnessDelta = 32 ;
  batchOpts.distortOpts.contrastLower = 0.5 ;
  batchOpts.distortOpts.contrastUpper = 1.5 ;
  batchOpts.distortOpts.hueDelta = 18 ;
  batchOpts.distortOpts.saturationLower = 0.5 ;
  batchOpts.distortOpts.saturationUpper = 1.5 ;
  batchOpts.distortOpts.randomOrderProb = 0 ;
  batchOpts.debug = opts.debug ;

  batchOpts.numThreads = 2 ;
  batchOpts.prefetch = true ;
  batchOpts.useGpu = numel(opts.train.gpus) >  0 ;
  batchOpts.resizers = {'bilinear', 'box', 'nearest', 'bicubic', 'lanczos2'} ;

  % configure paths
  expName = getExpNameFRCNN(opts.modelOpts, opts.dataOpts) ;
  if opts.debug, expName = [expName '-debug'] ; end
  expDir = fullfile(vl_rootnn, 'data', opts.dataOpts.name, expName) ;
  imdbTail = fullfile(opts.dataOpts.name, '/standard_imdb/imdb.mat') ;
  opts.dataOpts.imdbPath = fullfile(vl_rootnn, 'data', imdbTail) ;
  modelName = sprintf('local-%s-%s-%%d.mat', opts.modelOpts.type, ...
                                                   opts.dataOpts.name) ;
  opts.modelOpts.deployPath = fullfile(expDir, 'deployed', modelName) ;

  % configure meta options
  opts.batchOpts = batchOpts ;
  opts.eval_func = @faster_rcnn_pascal_evaluation ;
  faster_rcnn_train(expDir, opts) ;

% ---------------------------------------------------
function [opts, imdb] = prepareImdb(imdb, opts)
% ---------------------------------------------------
% set path to VOC 2007 devkit directory

  switch opts.dataOpts.trainData
    case '07', imdb.images.set(imdb.images.year == 2012) = -1 ;
    case '12', imdb.images.set(imdb.images.year == 2007) = -1 ;
    case '0712' % do nothing
    otherwise, error('Data %s not recognized', opts.dataOpts.trainData) ;
  end

  opts.train.val = find(imdb.images.set == 2) ;
  if opts.dataOpts.useValForTraining
    opts.train.train = find(imdb.images.set == 2 | imdb.images.set == 1) ;
  end
