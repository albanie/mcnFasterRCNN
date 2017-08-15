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
%   `gpus` :: []
%    If provided, the gpu ids to be used for processing.
%
%   `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%    The path to the directory containing the coco data
%
%   `pruneCheckpoints` :: true
%    Determines whether intermediate training files should be cleared to save
%    space after the training run has completed.
%
%   `nms` :: 'cpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `architecture` :: 'vgg16'
%    The trunk architecture used to initialise the model.
%
%   `useValForTraining` :: true
%    Whether the validation set (as defined in the original challenge) should
%    be included in the training set.
%
%   `flipAugmentation` :: true
%    Whether flipped images should be used in the training procedure.
%
%   `zoomAugmentation` :: false 
%    [REQUIRES mcnSSD install] - use SSD-style "zoom" augmentation to improve
%    performance
%
%   `patchAugmentation` :: false 
%    [REQUIRES mcnSSD install] - use SSD-style "patch" augmentation to improve
%    performance
%
%   `distortAugmentation` :: false 
%    Use SSD-style "distortion" augmentation to improve performance
%
%   `use_vl_imreadjpeg` :: true 
%    Use asynchronous image loader (slightly improves speed)
%
%   `zoomScale` :: 2 
%    Zoom magnitude used by SSD-style data augmentation
%
%   `confirmConfig` :: true 
%    Ask the user to confirm the experimental settings before running the 
%    experiment
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 3 ;
  opts.debug = 0 ; 
  opts.nms = 'gpu' ; % set to CPU if mcnNMS module is not installed
  opts.continue = 1 ;
  opts.confirmConfig = 0 ;
  opts.architecture = 'vgg16' ;
  opts.pruneCheckpoints = true ;
  opts.flipAugmentation = true ;
  opts.zoomAugmentation = false ;
  opts.useValForTraining = true ; 
  opts.patchAugmentation = false ;
  opts.use_vl_imreadjpeg = true ; 
  opts.distortAugmentation = false ;
  opts.zoomScale = 2 ;
  opts = vl_argparse(opts, varargin) ;

  % configure training options
  train.batchSize = numel(opts.gpus) ;
  train.derOutputs = 1 ; % give each loss the same weight
  train.gpus = opts.gpus ;
  train.numSubBatches = numel(train.gpus) ; %ceil(4 / max(numel(train.gpus), 1)) ;
  train.continue = opts.continue ;
  train.parameterServer.method = 'mmap' ;
  train.stats = {'rpn_loss_cls', 'rpn_loss_bbox','loss_cls', 'loss_bbox', ...
                        'multitask_loss'} ; % train end to end

  % configure dataset options
  dataOpts.name = 'pascal' ;
  dataOpts.trainData = '07' ; dataOpts.testData = '07' ;
  dataOpts.flipAugmentation = opts.flipAugmentation ;
  dataOpts.zoomAugmentation = opts.zoomAugmentation ;
  dataOpts.patchAugmentation = opts.patchAugmentation ;
  dataOpts.distortAugmentation = opts.distortAugmentation ;
  dataOpts.useValForTraining = opts.useValForTraining ;
  dataOpts.zoomScale = opts.zoomScale ;
  dataOpts.getImdb = @getPascalImdb ;
  dataOpts.prepareImdb = @prepareImdb ;
  dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;

  % configure model options
  modelOpts.type = 'faster-rcnn' ;
  modelOpts.nms = opts.nms ; 
  modelOpts.locWeight = 1 ;
  modelOpts.numClasses = 21 ;
  modelOpts.featStride = 16 ;
  modelOpts.ratios = [0.5, 1, 2] ;
  modelOpts.scales = [8, 16, 32] ;
  modelOpts.clipPriors = false ;
  modelOpts.net_init = @faster_rcnn_init ;
  modelOpts.deploy_func = @faster_rcnn_deploy ;
  modelOpts.batchSize = train.batchSize ;
  modelOpts.get_batch = @faster_rcnn_train_get_batch ;
  modelOpts.architecture = opts.architecture ;
  modelOpts.batchNormalization = false ;
  modelOpts.batchRenormalization = false ;
  modelOpts.CudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
  modelOpts.initMethod = 'gaussian' ;
  modelOpts.freezeBnorm = 0 ;
  modelOpts.mergeBnorm = 0 ;
  protoName = sprintf('%s_train.prototxt', opts.architecture) ;
  protoDir = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/misc') ;
  modelOpts.protoPath = fullfile(protoDir, protoName) ;

  % Set learning rates
  steadyLR = 0.001 ;
  gentleLR = 0.0001 ; 
  vGentleLR = 0.00001 ;

  if ~dataOpts.zoomAugmentation 
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
  train.learningRate = [steady gentle veryGentle] ;
  train.numEpochs = numel(train.learningRate) ;

  % configure batch opts
  batchOpts.scale = 600 ;
  batchOpts.maxScale = 1000 ;
  batchOpts.clipTargets = true ;
  batchOpts.patchOpts.use = dataOpts.patchAugmentation ;
  batchOpts.patchOpts.numTrials = 50 ;
  batchOpts.patchOpts.minPatchScale = 0.3 ;
  batchOpts.patchOpts.maxPatchScale = 1 ;
  batchOpts.patchOpts.minAspect = 0.5 ;
  batchOpts.patchOpts.maxAspect = 2 ;
  batchOpts.patchOpts.clipTargets = batchOpts.clipTargets ;

  batchOpts.flipOpts.use = dataOpts.flipAugmentation ;
  batchOpts.flipOpts.prob = 0.5 ;
  batchOpts.zoomOpts.use = dataOpts.zoomAugmentation ;
  batchOpts.zoomOpts.prob = 0.5 ;
  batchOpts.zoomOpts.minScale = 1 ;
  batchOpts.zoomOpts.maxScale = dataOpts.zoomScale ;

  batchOpts.distortOpts.use = dataOpts.distortAugmentation ;
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
  batchOpts.prefetch = false ; 
  batchOpts.useGpu = numel(train.gpus) >  0 ;
  batchOpts.use_vl_imreadjpeg = opts.use_vl_imreadjpeg ;
  batchOpts.resizers = {'bilinear', 'box', 'nearest', 'bicubic', 'lanczos2'} ;

  % configure paths
  expName = getExpNameFRCNN(modelOpts, dataOpts) ;
  if opts.debug, expName = [expName '-debug'] ; end
  expDir = fullfile(vl_rootnn, 'data', dataOpts.name, expName) ;
  imdbTail = fullfile(dataOpts.name, '/standard_imdb/imdb.mat') ;
  dataOpts.imdbPath = fullfile(vl_rootnn, 'data', imdbTail);
  modelName = sprintf('local-%s-%s-%%d.mat', modelOpts.type, dataOpts.name) ;
  modelOpts.deployPath = fullfile(expDir, 'deployed', modelName) ;

  % configure meta options
  opts.train = train ;
  opts.dataOpts = dataOpts ;
  opts.modelOpts = modelOpts ;
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
