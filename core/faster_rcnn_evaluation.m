function results = faster_rcnn_evaluation(expDir, net, opts)

% ----------------------------------------------------------------
%                                                     Prepare imdb
% ----------------------------------------------------------------
if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
else
    imdb = opts.dataOpts.getImdb(opts) ;
    imdbDir = fileparts(opts.dataOpts.imdbPath) ;
    if ~exist(imdbDir, 'dir'), mkdir(imdbDir) ; end
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
end

[opts, imdb] = opts.dataOpts.configureImdbOpts(expDir, opts, imdb) ;

switch opts.testset
    case 'train'
        setLabel = 1 ;
    case 'val'
        setLabel = 2 ;
    case 'test'
        setLabel = 3 ;
    case 'test-dev'
        setLabel = 4 ;
end

testIdx = find(imdb.images.set == setLabel) ;

% retrieve results from cache if possible
results = checkCache(opts, net, imdb, testIdx) ;
opts.dataOpts.displayResults(opts.modelName, results, opts) ;

% -------------------------------------------------
function res = checkCache(opts, net, imdb, testIdx)
% -------------------------------------------------
path = opts.cacheOpts.resultsCache ;

if exist(path, 'file') && ~opts.cacheOpts.refreshCache
  fprintf('loading results from cache\n') ;
  tmp = load(path) ; res = tmp.results ;
else
  p = computePredictions(net, imdb, testIdx, opts) ;
  decoded = decodePredictions(p, imdb, testIdx, opts) ;
  s.results = opts.dataOpts.eval_func(opts.modelName, decoded, imdb, opts) ;
  fprintf('saving to %s\n', path) ; save(path, '-struct', 's', '-v7.3') ; 
  res = s.results ;
end


% -------------------------------------------------------------------------
function decodedPreds = decodePredictions(p, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
numClasses = numel(imdb.meta.classes) ; 
% gather predictions by class, and store the corresponding 
% image id (i.e. image name) and bouding boxes
imageIds = cell(1, numClasses) ;
scores = cell(1, numClasses) ;
bboxes = cell(1, numClasses) ;

cPreds = p.cPreds ; 
bPreds = p.bPreds ; 
rois = p.rois ; 

for c = 1:numClasses - 1 % don't store bg 
  fprintf('extracting predictions for %s\n', imdb.meta.classes{c}) ;
  for p = 1:numel(testIdx)

    target = c + 1 ; % add offset for bg class

    % find predictions for current image
    cPreds_ = cPreds(target,:,p) ; 
    bPreds_ = bPreds((target-1)*4+1:target*4,:,p)' ; rois_ = rois(:,:,p)' ;
    cboxes = bbox_transform_inv(rois_, bPreds_);
    cls_dets = [cboxes cPreds_'] ;
    keep = bbox_nms(cls_dets, opts.modelOpts.nmsThresh) ;
    cls_dets = cls_dets(keep, :) ;
    sel_boxes = find(cls_dets(:,end) >= opts.modelOpts.confThresh) ;


    % clip predictions to fall in image and scale the 
    % bounding boxes from [0,1] to absolute pixel values
    if sel_boxes
      imsz = single(imdb.images.imageSizes{testIdx(p)}) ;
      minScaleFactor = opts.batchOpts.scale ./ min(imsz) ;
      pBoxes = cls_dets(sel_boxes,1:4) / minScaleFactor ; 
      pScores = cls_dets(sel_boxes,5) ;

      % clip predicitons
      pBoxes = min(max(pBoxes, 0), repmat(imsz([2 1]), size(pBoxes,1), 2)) ;

      switch opts.dataOpts.resultsFormat
        case 'minMax'
          ; % do nothing
        case 'minWH'
          pBoxes = [ pBoxes(1:2) pBoxes(3:4) - pBoxes(1:2) ] ;
        otherwise
          error('format %s not recognised', opts.dataOpts.resultsFormat) ;
      end

      % store results
      pId = imdb.images.name{testIdx(p)} ;
      scores{c} = vertcat(scores{c}, pScores) ;
      bboxes{c} = vertcat(bboxes{c}, pBoxes) ;
      imageIds{c} = vertcat(imageIds{c}, repmat({pId}, size(pScores))) ; 
    end

    if mod(p,100) == 1, fprintf('extracting %d/%d\n', p, numel(testIdx)) ; end
  end
end

decodedPreds.imageIds = imageIds ;
decodedPreds.scores = scores ;
decodedPreds.bboxes = bboxes ;

% -------------------------------------------------------
function p = computePredictions(net, imdb, testIdx, opts) 
% -------------------------------------------------------

prepareGPUs(opts, true) ;

p = struct() ;
params.testIdx = testIdx ;

if numel(opts.gpus) <= 1
   state = processDetections(net, imdb, params, opts) ;
   p.cPreds = state.clsPreds ; p.bPreds = state.bboxPreds ;
   p.rois = state.rois ;
else
  topK = opts.modelOpts.maxPreds ; numClasses = opts.modelOpts.numClasses ;
  p.clsPreds = zeros(numClasses, topK, numel(testIdx), 'single') ; 
  p.bboxPreds = zeros(4 * numClasses, topK, numel(testIdx), 'single') ; 
  p.rois = zeros(4, topK, numel(testIdx), 'single') ; 
  startup ;  % fix for parallel oddities
  spmd
   state = processDetections(net, imdb, params, opts) ;
  end
  for i = 1:numel(opts.gpus)
    state_ = state{i} ;
    p.cPreds(:,:,state_.computedIdx) = state_.clsPreds ;
    p.bPreds(:,:,state_.computedIdx) = state_.bboxPreds ;
    p.rois(:,:,state_.computedIdx) = state_.rois ;
  end
end

% -------------------------------------------------------------------
function state = processDetections(net, imdb, params, opts, varargin) 
% -------------------------------------------------------------------

sopts.scale = [] ;
sopts = vl_argparse(sopts, varargin) ;

% benchmark speed
num = 0 ; adjustTime = 0 ; stats.time = 0 ;
stats.num = num ; start = tic ; testIdx = params.testIdx ;
%clsIdx = net.getVarIndex('cls_prob') ;
%bboxIdx = net.getVarIndex('bbox_pred') ;
%%roisIdx = net.getVarIndex('proposal') ;
%roisIdx = net.getVarIndex('rois') ;
if ~isempty(opts.gpus), net.move('gpu') ; end


% pre-compute the indices of the predictions made by each worker
startIdx = labindex:numlabs:opts.batchOpts.batchSize ;
idx = arrayfun(@(x) {x:opts.batchOpts.batchSize:numel(testIdx)}, startIdx) ;
computedIdx = sort(horzcat(idx{:})) ;

%net.mode = 'test' ;

% only the top K preds kept
topK = opts.modelOpts.maxPreds ; numClasses = opts.modelOpts.numClasses ;
state.clsPreds = zeros(numClasses, topK, numel(computedIdx), 'single') ; 
state.bboxPreds = zeros(4 * numClasses, topK, numel(computedIdx), 'single') ; 
state.rois = zeros(4, topK, numel(computedIdx), 'single') ; 
state.computedIdx = computedIdx ; offset = 1 ;

for t = 1:opts.batchOpts.batchSize:numel(testIdx) 
  progress = fix((t-1) / opts.batchOpts.batchSize) + 1 ; % display progress
  totalBatches = ceil(numel(testIdx) / opts.batchOpts.batchSize) ;
  fprintf('evaluating batch %d / %d: ', progress, totalBatches) ;
  batchSize = min(opts.batchOpts.batchSize, numel(testIdx) - t + 1) ;
  batchStart = t + (labindex - 1) ;
  batchEnd = min(t + opts.batchOpts.batchSize - 1, numel(testIdx)) ;
  batch = testIdx(batchStart : numlabs : batchEnd) ;
  num = num + numel(batch) ;
  %fprintf('pre-batch check\n') ; drawnow('update') ;
  if numel(batch) == 0, continue ; end
  %fprintf('post-batch check\n') ; drawnow('update') ;
  args = {imdb, batch, opts} ;
  if ~isempty(sopts.scale), args = {args{:}, sopts.scale} ; end 
  %fprintf('post-args check\n') ; drawnow('update') ;
  %fprintf('processing %d\n', batch(1)) ; drawnow('update') ;
  %disp('args:') ; disp(args) ; drawnow('update') ;
  inputs = opts.modelOpts.get_eval_batch(args{:}) ;

  if opts.prefetch
    batchStart_ = t + (labindex - 1) + opts.batchOpts.batchSize ;
    batchEnd_ = min(t + 2*opts.batchOpts.batchSize - 1, numel(testIdx)) ;
    nextBatch = testIdx(batchStart_: numlabs : batchEnd_) ;
    args = {imdb, nextBatch, opts} ;
    if ~isempty(sopts.scale), args = {args{:}, sopts.scale} ; end 
    opts.modelOpts.get_eval_batch(args{:}, 'prefetch', true) ;
  end

  %net.setInputs('data', inputs{2}) ; 
  %fprintf('pre-eval\n') ; drawnow('update') ;
  net.eval(inputs, 'forward') ;
  %net.eval(inputs) ;
  %fprintf('post-eval\n') ; drawnow('update') ;

  storeIdx = offset:offset + numel(batch) - 1 ; offset = offset + numel(batch) ;
  %out = net.vars([clsIdx bboxIdx roisIdx]) ; 
  %disp('out-value') ; disp(out) ; drawnow('update') ;
  %[cPreds, bPreds, rois] = out{:} ;
  %disp('unpacking completed') ; drawnow('update') ;
  %cPreds = net.vars(clsIdx).value; 
  %bPreds = net.vars(bboxIdx).value ; 
  %rois = net.vars(roisIdx).value ; 
  cPreds = net.getValue('cls_prob'); 
  bPreds = net.getValue('bbox_pred') ; 
  rois = net.getValue('proposal') ; 
  %disp('picked the values') ; drawnow('update') ;
  state.clsPreds(:,1:size(cPreds,4),storeIdx) = gather(squeeze(cPreds)) ;
  state.bboxPreds(:,1:size(bPreds,4),storeIdx) = gather(squeeze(bPreds)) ;
  state.rois(:,1:size(rois,2),storeIdx) = gather(rois(2:end,:)) ;
  time = toc(start) + adjustTime ; batchTime = time - stats.time ;
  stats.num = num ; stats.time = time ; currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  %fprintf('post2-eval\n') ; drawnow('update') ;

  if t == 3*opts.batchOpts.batchSize + 1
      % compensate for the first three iterations, which are outliers
      adjustTime = 4*batchTime - time ; stats.time = time + adjustTime ;
  end
  fprintf('speed %.1f (%.1f) Hz', averageSpeed, currentSpeed) ; fprintf('\n') ;
end

net.move('cpu') ;

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end
