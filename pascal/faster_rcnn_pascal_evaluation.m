function [aps, speed] = faster_rcnn_pascal_evaluation(varargin)
%FASTER_RCNN_EVALUATE  Evaluate a trained Faster-RCNN model on 
% PASCAL VOC 2007

% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.
opts.net = [] ;
opts.modelName = 'faster-rcnn-vggvd-pascal' ;
opts.evalVersion = 'fast' ;
opts.expDir = '' ; % preserve interface
opts.optsStruct = struct() ; 
opts.gpus = [2 3] ;
opts.refreshCache = true ;
opts.dataRoot = fullfile(vl_rootnn, 'data/datasets') ;
opts = vl_argparse(opts, varargin) ;

% load network and convert to autonn
if isempty(opts.net), opts.net = faster_rcnn_zoo(opts.modelName) ; end
%layers = Layer.fromDagNN(opts.net, @faster_rcnn_autonn_custom_fn) ;
%opts.net = Net(layers{:}) ;

% evaluation options
opts.testset = 'test' ; 
opts.prefetch = false ;

% configure batch opts
batchOpts.batchSize = numel(opts.gpus) * 1 ;
batchOpts.numThreads = numel(opts.gpus) * 4 ;
batchOpts.use_vl_imreadjpeg = 1 ; 
batchOpts.maxScale = 1000 ;
batchOpts.scale = 600 ;
batchOpts.averageImage = opts.net.meta.normalization.averageImage ;

% cache configuration 
cacheOpts.refreshCache = opts.refreshCache ;

% configure model options
modelOpts.get_eval_batch = @faster_rcnn_eval_get_batch ;
modelOpts.maxPredsPerClass = 100 ; 
modelOpts.maxPreds = 300 ; % the maximum number of total preds/img
modelOpts.numClasses = numel(opts.net.meta.classes.name) ;
modelOpts.nmsThresh = 0.3 ;
modelOpts.confThresh = 0.1 ;

% configure dataset options
dataOpts.name = 'pascal' ;
dataOpts.resultsFormat = 'minMax' ; 
dataOpts.getImdb = @getPascalImdb ;
dataOpts.dataRoot = opts.dataRoot ;
dataOpts.eval_func = @pascal_eval_func ;
dataOpts.evalVersion = opts.evalVersion ;
dataOpts.displayResults = @displayPascalResults ;
dataOpts.configureImdbOpts = @configureImdbOpts ;
dataOpts.imdbPath = fullfile(vl_rootnn, 'data/pascal/standard_imdb/imdb.mat') ;

% configure paths
expDir = fullfile(vl_rootnn, 'data/evaluations', dataOpts.name, opts.modelName) ;
resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
evalCacheDir = fullfile(expDir, 'eval_cache') ;
cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
cacheOpts.evalCacheDir = evalCacheDir ;

if ~exist(evalCacheDir, 'dir'), 
    mkdir(evalCacheDir) ;
    mkdir(fullfile(evalCacheDir, 'cache')) ;
end

% configure meta options
opts.dataOpts = dataOpts ;
opts.modelOpts = modelOpts ;
opts.batchOpts = batchOpts ;
opts.cacheOpts = cacheOpts ;

results = faster_rcnn_evaluation(expDir, opts.net, opts) ;

% ------------------------------------------------------------------
function aps = pascal_eval_func(modelName, decodedPreds, imdb, opts)
% ------------------------------------------------------------------

numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
aps = zeros(numClasses, 1) ;

for c = 1:numClasses
    className = imdb.meta.classes{c + 1} ; % offset for background
    results = eval_voc(className, ...
                       decodedPreds.imageIds{c}, ...
                       decodedPreds.bboxes{c}, ...
                       decodedPreds.scores{c}, ...
                       opts.dataOpts.VOCopts, ...
                       'evalVersion', opts.dataOpts.evalVersion) ;
    fprintf('%s %.1\n', className, 100 * results.ap_auc) ;
    aps(c) = results.ap_auc ; 
end
save(opts.cacheOpts.resultsCache, 'aps') ;

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(expDir, opts, imdb)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)

if 0  % benchmark
  keep = 50 ; testIdx = find(imdb.images.set == 3) ;
  imdb.images.set(testIdx(keep:end)) = 4 ;
end
opts.dataOpts = configureVOC(expDir, opts.dataOpts, 'test') ;

%-----------------------------------------------------------
function dataOpts = configureVOC(expDir, dataOpts, testset) 
%-----------------------------------------------------------
% LOADPASCALOPTS Load the pascal VOC database options
%
% NOTE: The Pascal VOC dataset has a number of directories 
% and attributes. The paths to these directories are 
% set using the VOCdevkit code. The VOCdevkit initialization 
% code assumes it is being run from the devkit root folder, 
% so we make a note of our current directory, change to the 
% devkit root, initialize the pascal options and then change
% back into our original directory 

VOCRoot = fullfile(dataOpts.dataRoot, 'VOCdevkit2007') ;
VOCopts.devkitCode = fullfile(VOCRoot, 'VOCcode') ;

% check the existence of the required folders
assert(logical(exist(VOCRoot, 'dir')), 'VOC root directory not found') ;
assert(logical(exist(VOCopts.devkitCode, 'dir')), 'devkit code not found') ;

currentDir = pwd ;
cd(VOCRoot) ;
addpath(VOCopts.devkitCode) ;

% VOCinit loads database options into a variable called VOCopts
VOCinit ; 

dataDir = fullfile(VOCRoot, '2007') ;
VOCopts.localdir = fullfile(dataDir, 'local') ;
VOCopts.imgsetpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.imgpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.annopath = fullfile(dataDir, 'Annotations/%s.xml') ;
VOCopts.cacheDir = fullfile(expDir, '2007/Results/Cache') ;
VOCopts.drawAPCurve = false ;
VOCopts.testset = testset ;
detDir = fullfile(expDir, 'VOCdetections') ;

% create detection and cache directories if required
requiredDirs = {VOCopts.localdir, VOCopts.cacheDir, detDir} ;
for i = 1:numel(requiredDirs)
    reqDir = requiredDirs{i} ;
    if ~exist(reqDir, 'dir') 
        mkdir(reqDir) ;
    end
end

VOCopts.detrespath = fullfile(detDir, sprintf('%%s_det_%s_%%s.txt', 'test')) ;
dataOpts.VOCopts = VOCopts ;

% return to original directory
cd(currentDir) ;

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------

fprintf('\n============\n') ;
fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
fprintf('\n============\n') ;

% display all relevant experiments
cacheRoot = fileparts(fileparts(opts.cacheOpts.evalCacheDir)) ;
cacheDirs = extractCacheDirs(cacheRoot) ;

% move current experiment to head
pos = find(strcmp(cacheDirs, opts.cacheOpts.evalCacheDir)) ;
tmp = cacheDirs{1} ; cacheDirs{1} = cacheDirs{pos} ; cacheDirs{pos} = tmp ;

% save expTag (this lets us have multiple tags per experiment)
meta = struct() ;
if ~isfield(opts.net.meta, 'expTag')
  opts.net.meta.expTag = 'all' ;
end
path = fullfile(opts.cacheOpts.evalCacheDir, opts.net.meta.expTag) ;
if ~exist(fileparts(path), 'dir')
  mkdir(fileparts(path)) ;
end
save(path, '-struct', 'meta') ;
printPascalResults({opts.cacheOpts.evalCacheDir}, 'orientation', 'portrait') ;

% ------------------------------------------
function cacheDirs = extractCacheDirs(root)
% ------------------------------------------
files = ignoreSystemFiles(dir(fullfile(root, '*'))) ;
names = {files.name} ;
cacheDirs = cellfun(@(x) {fullfile(root, x, 'eval_cache')}, names) ;
