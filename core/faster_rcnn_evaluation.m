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
keyboard

for t = 1:numel(testIdx)

  imsz = single(imdb.images.imageSizes{testIdx(t)}) ;
  maxSc = opts.batchOpts.maxScale ; 
  factor = max(opts.batchOpts.scale ./ imsz) ; 
  if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end

  % find predictions for current image
  cPreds_ = cPreds(:,:,t) ; 
  bPreds_ = bPreds(:,:,t)' ; 

  rois_ = rois(:,:,t)' ; 
  keep = find(rois_(:,4) ~= 0) ; % drop unused RoIs
  rois_ = rois_(keep,:) ; 
  bPreds_ = bPreds_(keep,:) ; 
  cPreds_ = cPreds_(:,keep) ;
  rois_ = rois_ - 1 ; % undo offset required by roipool
  boxes = rois_ / factor ;
  cBoxes = bboxTransformInv(boxes, bPreds_) ;
  cBoxes = clipBoxes(cBoxes, imsz) ;

  numKept = 0 ;

  for c = 1:numClasses - 1 % don't store bg 
    target = c + 1 ; % add offset for bg class

    % compute regressed proposals
    tBoxes = cBoxes(:,(target-1)*4+1:(target)*4) ;
    tScores = cPreds_(target,:)' ;
    cls_dets = [tBoxes tScores] ;
    
    % drop preds below threshold
    keep = find(cls_dets(:,end) >= opts.modelOpts.confThresh) ;
    cls_dets = cls_dets(keep,:) ;
    if ~numel(keep), continue ; end

    % TODO(samuel): Move last round of NMS into the computePredictions 
    % function for a fair timing benchmark 
    % heuristic: keep a fixed number of dets per class per image before nms
    [~,si] = sort(cls_dets(:,5),'descend') ;
    cls_dets = cls_dets(si,:) ;
    numKeep = min(size(cls_dets,1),opts.modelOpts.maxPredsPerImage) ;
    cls_dets = cls_dets(1:numKeep,:) ;

    keep = bbox_nms(cls_dets, opts.modelOpts.nmsThresh) ;
    cls_dets = cls_dets(keep, :) ;

    if numel(keep)
      numKept = numKept + numel(keep) ;
      pBoxes = cls_dets(:,1:4) + 1 ; % fix offset
      pScores = cls_dets(:,5) ;
      switch opts.dataOpts.resultsFormat
        case 'minMax'
          % do nothing
        case 'minWH'
          pBoxes = [ pBoxes(:, 1:2) pBoxes(:,3:4) - pBoxes(:,1:2) ] ;
        otherwise
          error('format %s not recognised', opts.dataOpts.resultsFormat) ;
      end

      % store results
      pId = imdb.images.name{testIdx(t)} ;
      scores{c} = vertcat(scores{c}, pScores) ;
      bboxes{c} = vertcat(bboxes{c}, pBoxes) ;
      imageIds{c} = vertcat(imageIds{c}, repmat({pId}, size(pScores))) ; 
    end
  end

  %if numKept > opts.modelOpts.maxPredsPerImage
    %keyboard
  %end
  if mod(t,100) == 1, fprintf('extracting %d/%d\n', t, numel(testIdx)) ; end
end

% cheat:
%testIdx = find(imdb.images.set == 3) ;
if 0 
  caffePreds = load('/tmp/stored_dets.mat') ;
  useCaffe = 0 ;
  if useCaffe 
    zz = load(fullfile(vl_rootnn,'data/pascal/standard_imdb/imdb.mat')) ;
    %testIdx = find(zz.images.set == 3) ;
  end

  keep = 1:numel(testIdx) ;

  caffeAll = caffePreds.all_boxes(:,keep) ;
  caffeIds = cell(size(imageIds)) ;
  caffeBoxes = cell(size(bboxes)) ;
  caffeScores = cell(size(scores)) ;

  for c = 1:numClasses - 1
    caffePreds_ = caffeAll(c+1,:) ;
    caffeIds_ = cellfun(@(x,y) {repmat({imdb.images.name{y}}, size(x,1), 1)}, caffePreds_, num2cell(testIdx)) ;
    scoredBoxes = vertcat(caffePreds_{:}) ;
    caffeIds{c} = vertcat(caffeIds_{:}) ; 
    caffeBoxes{c} = scoredBoxes(:,1:4) ;
    caffeScores{c} = scoredBoxes(:,5) ;
    % tmp
    cb = bboxCoder(caffeBoxes{c}, 'MinMax', 'MinWH') ;
    mb = bboxCoder(bboxes{c}, 'MinMax', 'MinWH') ;
    o = bboxOverlapRatio(cb, mb) ;
    imageIds_ = imageIds{c} ;
    numC = size(caffeIds{c}, 1) ;  numM = size(imageIds_, 1) ;
    fprintf('caffe: %d vs mcn: %d\n', numC, numM) ;

    %if numC ~= size(imageIds_, 1)
      %keyboard 
    %end

  end

  if useCaffe
    imageIds = caffeIds ;
    bboxes = caffeBoxes ;
    scores = caffeScores ;
  end
end

decodedPreds.imageIds = imageIds ;
decodedPreds.scores = scores ;
decodedPreds.bboxes = bboxes ;

% ------------------------------------------------
function predBoxes = bboxTransformInv(boxes, deltas)
% ------------------------------------------------
W = boxes(:,3) - boxes(:,1) + 1 ;
H = boxes(:,4) - boxes(:,2) + 1 ;
ctrX = boxes(:,1) + 0.5 * W ;
ctrY = boxes(:,2) + 0.5 * H ;

dX = deltas(:,1:4:end) ; dY = deltas(:,2:4:end) ;
dW = deltas(:,3:4:end) ; dH = deltas(:,4:4:end) ;
predCtrX = bsxfun(@plus, bsxfun(@times, dX,W), ctrX) ;
predCtrY = bsxfun(@plus, bsxfun(@times, dY,H), ctrY) ;
predW = bsxfun(@times, exp(dW), W) ;
predH = bsxfun(@times, exp(dH), H) ;

predBoxes = zeros(size(deltas)) ;
predBoxes(:,1:4:end) = predCtrX - 0.5 * predW ;
predBoxes(:,2:4:end) = predCtrY - 0.5 * predH ;
predBoxes(:,3:4:end) = predCtrX + 0.5 * predW ;
predBoxes(:,4:4:end) = predCtrY + 0.5 * predH ;

% -------------------------------------
function boxes = clipBoxes(boxes, imsz)
% -------------------------------------
boxes(:,1:4:end) = max(min(boxes(:,1:4:end),imsz(2)-1),0);
boxes(:,2:4:end) = max(min(boxes(:,2:4:end),imsz(1)-1),0);
boxes(:,3:4:end) = max(min(boxes(:,3:4:end),imsz(2)-1),0);
boxes(:,4:4:end) = max(min(boxes(:,4:4:end),imsz(1)-1),0);

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
if ~isempty(opts.gpus), net.move('gpu') ; end

% pre-compute the indices of the predictions made by each worker
startIdx = labindex:numlabs:opts.batchOpts.batchSize ;
idx = arrayfun(@(x) {x:opts.batchOpts.batchSize:numel(testIdx)}, startIdx) ;
computedIdx = sort(horzcat(idx{:})) ;

% only the top K preds kept
topK = opts.modelOpts.maxPreds ; 
numClasses = opts.modelOpts.numClasses ;

state.clsPreds = zeros(numClasses, topK, numel(computedIdx), 'single') ; 
state.bboxPreds = zeros(4 * numClasses, topK, numel(computedIdx), 'single') ; 
state.rois = zeros(4, topK, numel(computedIdx), 'single') ; 
state.computedIdx = computedIdx ; 

offset = 1 ;

for t = 1:opts.batchOpts.batchSize:numel(testIdx) 
  progress = fix((t-1) / opts.batchOpts.batchSize) + 1 ; % display progress
  totalBatches = ceil(numel(testIdx) / opts.batchOpts.batchSize) ;
  fprintf('evaluating batch %d / %d: ', progress, totalBatches) ;
  batchSize = min(opts.batchOpts.batchSize, numel(testIdx) - t + 1) ;
  batchStart = t + (labindex - 1) ;
  batchEnd = min(t + opts.batchOpts.batchSize - 1, numel(testIdx)) ;
  batch = testIdx(batchStart : numlabs : batchEnd) ;
  num = num + numel(batch) ;
  if numel(batch) == 0, continue ; end
  args = {imdb, batch, opts} ;
  if ~isempty(sopts.scale), args = [args, {sopts.scale}] ; end 
  inputs = opts.modelOpts.get_eval_batch(args{:}) ;

  if opts.prefetch
    batchStart_ = t + (labindex - 1) + opts.batchOpts.batchSize ;
    batchEnd_ = min(t + 2*opts.batchOpts.batchSize - 1, numel(testIdx)) ;
    nextBatch = testIdx(batchStart_: numlabs : batchEnd_) ;
    args = {imdb, nextBatch, opts} ;
    if ~isempty(sopts.scale), args = [args, {sopts.scale}] ; end 
    opts.modelOpts.get_eval_batch(args{:}, 'prefetch', true) ;
  end

  net.eval(inputs, 'forward') ;
  storeIdx = offset:offset + numel(batch) - 1 ; offset = offset + numel(batch) ;
  cPreds = net.getValue('cls_prob'); 
  bPreds = net.getValue('bbox_pred') ; 
  rois = net.getValue('proposal') ; 

  state.clsPreds(:,1:size(cPreds,4),storeIdx) = gather(squeeze(cPreds)) ;
  state.bboxPreds(:,1:size(bPreds,4),storeIdx) = gather(squeeze(bPreds)) ;
  state.rois(:,1:size(rois,2),storeIdx) = gather(rois(2:end,:)) ;
  time = toc(start) + adjustTime ; batchTime = time - stats.time ;
  stats.num = num ; stats.time = time ; currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;

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
