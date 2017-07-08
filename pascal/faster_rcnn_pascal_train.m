function faster_rcnn_pascal_train(varargin)

opts.gpus = 3 ;
opts.debug = 0 ; 
opts.continue = true ;
opts.confirmConfig = false ;
opts.architecture = 'vgg16' ;
opts.pruneCheckpoints = true ;
opts.flipAugmentation = true ;
opts.distortAugmentation = false ;
opts.zoomAugmentation = false ;
opts.patchAugmentation = false ;
opts.use_vl_imreadjpeg = true ; 
opts = vl_argparse(opts, varargin) ;

% configure training options
train.batchSize = 1 ;
train.gpus = opts.gpus ;
train.continue = opts.continue ;
train.parameterServer.method = 'tmove' ;
train.numSubBatches = ceil(4 / max(numel(train.gpus), 1)) ;
train.derOutputs = {'multitask_loss', 1} ;

% configure dataset options
dataOpts.name = 'pascal' ;
dataOpts.trainData = '07' ; dataOpts.testData = '07' ;
dataOpts.flipAugmentation = opts.flipAugmentation ;
dataOpts.zoomAugmentation = opts.zoomAugmentation ;
dataOpts.patchAugmentation = opts.patchAugmentation ;
dataOpts.distortAugmentation = opts.distortAugmentation ;
dataOpts.useValForTraining = true ;
dataOpts.zoomScale = 4 ;
dataOpts.getImdb = @getPascalImdb ;
dataOpts.prepareImdb = @prepareImdb ;
dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;

% configure model options
modelOpts.type = 'faster_rcnn' ;
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


% Set learning rates
warmup = 0.0001 ;
steadyLR = 0.001 ;
gentleLR = 0.0001 ;
vGentleLR = 0.00001 ;

if dataOpts.zoomAugmentation
    numSteadyEpochs = 155 ;
    numGentleEpochs = 38 ;
    numVeryGentleEpochs = 38 ;
else
    numSteadyEpochs = 75 ;
    numGentleEpochs = 35 ;
    numVeryGentleEpochs = 0 ;
end

steady = steadyLR * ones(1, numSteadyEpochs) ;
gentle = gentleLR * ones(1, numGentleEpochs) ;
veryGentle = vGentleLR * ones(1, numVeryGentleEpochs) ;
train.learningRate = [warmup steady gentle veryGentle] ;
train.numEpochs = numel(train.learningRate) ;

% configure batch opts
batchOpts.clipTargets = false ;
batchOpts.scale = 600 ;
batchOpts.maxScale = 1000 ;
batchOpts.patchOpts.use = dataOpts.patchAugmentation ;
batchOpts.patchOpts.numTrials = 50 ;
batchOpts.patchOpts.minPatchScale = 0.3 ;
batchOpts.patchOpts.maxPatchScale = 1 ;
batchOpts.patchOpts.minAspect = 0.5 ;
batchOpts.patchOpts.maxAspect = 2 ;

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
expDir = fullfile(vl_rootnn, 'data', dataOpts.name, opts.architecture) ;
imdbTail = fullfile(dataOpts.name, '/standard_imdb/imdb.mat') ;
dataOpts.imdbPath = fullfile(vl_rootnn, 'data', imdbTail);
modelName = sprintf('local-%s-%s-%d-%%d.mat', modelOpts.type, dataOpts.name) ;
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
    case '07' % to restrict to 2007, remove training 2012 data
        imdb.images.set(imdb.images.year == 2012 & imdb.images.set == 1) = -1 ;
    case '12' % to restrict to 2012, remove 2007 training data
        imdb.images.set(imdb.images.year == 2007 & imdb.images.set == 1) = -1 ;
    case '0712' % do nothing
    otherwise
        error('Training data %s not recognized', opts.dataOpts.trainData) ;
end

opts.train.val = find(imdb.images.set == 2) ;
if opts.dataOpts.useValForTraining
    opts.train.train = find(imdb.images.set == 2 | imdb.images.set == 1) ;
end

if 1
  idx = find(strcmp('007054', imdb.images.name)) ;
  opts.train.train = idx ;
end
