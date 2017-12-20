function aps = faster_rcnn_pascal_evaluation(varargin)
%FASTER_RCNN_PASCAL_EVALUATION Evaluate a Faster-RCNN model on VOC 2007
%   FASTER_RCNN_PASCAL_EVALUATION computes and evaluates a set of detections
%   for a given Faster-RCNN detector on the Pascal VOC 2007 test set.
%
%   FASTER_RCNN_PASCAL_EVALUATION(..'name', value) accepts the following 
%   options:

%   `year` :: 2007
%    The year of the challenge to evalutate on. Currently 2007 (val and test)
%    and 2012 (val) are supported.  Predictions for 2012 test must be submitted
%    to the official evaluation server to obtain scores.
%
%   `net` :: []
%    The `autonn` network object to be evaluated.  If not supplied, a network
%    will be loaded instead by name from the detector zoo.

%   `testset` :: 'test'
%    The subset of pascal VOC 2007 to be used for evaluation.
%
%   `gpus` :: []
%    If provided, the gpu ids to be used for processing.
%
%   `evalVersion` :: 'fast'
%    The type of VOC evaluation code to be run.  The options are 'official', 
%    which runs the original (slow) pascal evaluation code, or 'fast', which
%    runs an optimised version which is useful during development.
%
%   `nms` :: 'cpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `modelName` :: 'faster-rcnn-vggvd-pascal'
%    The name of the detector to be evaluated (used to generate output
%    file names, caches etc.)
%
%   `refreshCache` :: false
%    If true, overwrite previous predictions by any detector sharing the 
%    same model name, otherwise, load results directly from cache.
%
% ----------------------------------------------------------------------------
%   `modelOpts` :: struct(...)
%    A structure of options relating to the properties of the model, with the 
%    following fields:
%      `predVar` :: 'detection_out'
%       The name of the output prediction variable of the network 
%
%      `maxPreds` :: 300 
%       The maximum number of predictions that are kept per image during 
%       inference.
%
%      `nmsThresh` :: 0.3 
%       The NMS threshold used to select predictions on a single image.
%
%      `confThresh` :: 0.0 
%       The minimum confidence required for a prediction to be scored as a 
%       "detection" by the network.

% ----------------------------------------------------------------------------
%   `dataOpts` :: struct(...)
%    A structure of options setting paths for the data, with the following 
%    fields:
%      `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%       The path to the directory containing the pascal data
%
%
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.net = [] ;
  opts.gpus = 4 ;
  opts.nms = 'cpu' ;  
  opts.year = 2007 ;
  opts.testset = 'test' ; 
  opts.evalVersion = 'fast' ;
  opts.modelName = 'faster-rcnn-mcn-vggvd-pascal' ;
  opts.refreshCache = false ;

  % configure dataset options
  opts.dataOpts.name = 'pascal' ;
  opts.dataOpts.resultsFormat = 'minMax' ; 
  opts.dataOpts.getImdb = @getCombinedPascalImdb ;
  opts.dataOpts.dataRoot = fullfile(vl_rootnn, 'data/datasets') ;
  opts.dataOpts.eval_func = @pascal_eval_func ;
  opts.dataOpts.evalVersion = opts.evalVersion ;
  opts.dataOpts.displayResults = @displayPascalResults ;
  opts.dataOpts.decoder = 'serial' ;
  opts.dataOpts.configureImdbOpts = @configureImdbOpts ;
  opts.dataOpts.imdbPath = fullfile(vl_rootnn, ...
                                    'data/pascal/standard_imdb/imdb.mat') ;

  % configure model options
  opts.modelOpts.maxPreds = 300 ; % the maximum number of total preds/img
  opts.modelOpts.nmsThresh = 0.3 ;
  opts.modelOpts.numClasses = 21 ; % includes background for pascal
  opts.modelOpts.confThresh = 0.0 ;
  opts.modelOpts.maxPredsPerImage = 100 ; 
  opts.modelOpts.classAgnosticReg = false ; 
  opts.modelOpts.get_eval_batch = @faster_rcnn_eval_get_batch ;

  % configure batch opts
  opts.batchOpts.scale = 600 ;
  opts.batchOpts.maxScale = 1000 ;
  opts.batchOpts.use_vl_imreadjpeg = 1 ;
  opts.batchOpts.numThreads = 4 ;
  opts.batchOpts.prefetch = true ;
  opts = vl_argparse(opts, varargin) ;

  % if needed, load network and convert to autonn
  if isempty(opts.net)
    opts.net = faster_rcnn_zoo(opts.modelName) ; 
    layers = Layer.fromDagNN(opts.net, @faster_rcnn_autonn_custom_fn) ;
    net = Net(layers{:}) ;
  else
    net = opts.net ;
  end

  net = configureNMS(net, opts) ; % configure NMS optimisations if required
  opts.batchOpts.batchSize = max(numel(opts.gpus) * 1, 1) ; % use bsize 1 on cpu
  opts.batchOpts.averageImage = net.meta.normalization.averageImage ;


  % configure paths and cache 
  expDir = fullfile(vl_rootnn, 'data/evaluations', opts.dataOpts.name, ...
                                                          opts.modelName) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, sprintf('eval_cache-%d', opts.year)) ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;
  cacheOpts.refreshCache = opts.refreshCache ;
  if ~exist(evalCacheDir, 'dir'), mkdir(evalCacheDir) ; end
  opts.cacheOpts = cacheOpts ;

  aps = faster_rcnn_evaluation(expDir, net, opts) ;

% ------------------------------------------------------------------
function aps = pascal_eval_func(modelName, decodedPreds, imdb, opts)
% ------------------------------------------------------------------
  fprintf('evaluating predictions for %s\n', modelName) ;
  numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
  aps = zeros(numClasses, 1) ;
  if (opts.year == 2012) && strcmp(opts.testset, 'test')
    fprintf('preds on 2012 test set must be submitted to the eval server\n') ;
    keyboard % TODO(samuel): Add support for output format
  else
    for c = 1:numClasses
      className = imdb.meta.classes{c + 1} ; % offset for background
      results = eval_voc(className, ...
                         decodedPreds.imageIds{c}, ...
                         decodedPreds.bboxes{c}, ...
                         decodedPreds.scores{c}, ...
                         opts.dataOpts.VOCopts, ...
                         'evalVersion', opts.dataOpts.evalVersion, ...
                         'year', opts.year) ;
      fprintf('%s %.1\n', className, 100 * results.ap) ;
      aps(c) = results.ap_auc ;
    end
    save(opts.cacheOpts.resultsCache, 'aps') ;
  end

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(expDir, opts, imdb)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)
  switch opts.year   
    case 2007, imdb.images.set(imdb.images.year == 2012) = -1 ;   
    case 2012, imdb.images.set(imdb.images.year == 2007) = -1 ;   
    case 0712 % do nothing    
    otherwise, error('Data from year %s not recognized', opts.year) ;    
  end   

  % ignore images that do not reside in the classification & detection challenge
  imdb.images.set(~imdb.images.classification) = -1 ;
  VOCopts = configureVOC(expDir, opts.dataOpts.dataRoot, opts.year) ;
  VOCopts.testset = opts.testset ;
  opts.dataOpts.VOCopts = VOCopts ;

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------
  fprintf('============\n') ;
  fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
  fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
  fprintf('============\n') ;

% ------------------------------------
function net = configureNMS(net, opts)
% ------------------------------------
%CONFIGURENMS - update porposal rpn to optimise performance
%  CONFIGURENMS(NET, OPTS) updates NMS to run on GPU (if specified)

  dnet = Layer.fromCompiledNet(net) ; % decompile
  cls_head = dnet{1} ; bbox_head = dnet{2} ;

  % update NMS
  prev = cls_head.find(@vl_nnproposalrpn, 1) ;
  in = [ prev.inputs {'nms', opts.nms}] ; % update nms option
  proposals = Layer.create(@vl_nnproposalrpn, in) ;
  proposals.name = prev.name ;
  cls_head.find(@vl_nnroipool, 1).inputs{2} = proposals ; % reattach

  net = Net(cls_head, bbox_head) ; % recompile
