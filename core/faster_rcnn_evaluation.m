function results = faster_rcnn_evaluation(expDir, net, opts)
%FASTER_RCNN_EVALUATION - run detector evaluation
%  FASTER_RCNN_EVALUATION(EXPDIR, NET) - evaluates the network NET
%  on the imdb specified (as a path option), and stores results in 
%  EXPDIR.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  % load/create imdb and configure
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
    case 'train', setLabel = 1 ;
    case 'val', setLabel = 2 ;
    case 'test', setLabel = 3 ;
    case 'test-dev', setLabel = 4 ;
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
    %p = load('/tmp/pred.mat') ;
    decoded = decodePredictions(p, imdb, testIdx, opts) ;
    s.results = opts.dataOpts.eval_func(opts.modelName, decoded, imdb, opts) ;
    fprintf('saving to %s\n', path) ; save(path, '-struct', 's', '-v7.3') ; 
    res = s.results ;
  end


% -------------------------------------------------------------------------
function decodedPreds = decodePredictions(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
  args = {predictions, imdb, testIdx, opts} ;
  switch opts.dataOpts.decoder 
    % For debuggin/small datasets serial decoding is useful
    case 'serial', decodedPreds = decodeSerial(args{:}) ;
    case 'custom', decodedPreds = opts.dataOpts.customDecoder(args{:}) ;
    otherwise, error('deocoder %s not recognised',opts.dataOpts.deocoder) ;
  end

% -------------------------------------------------------------------------
function decodedPreds = decodeSerial(p, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
  numClasses = numel(imdb.meta.classes) ; 
  imageIds = cell(1, numClasses) ;
  scores = cell(1, numClasses) ;
  bboxes = cell(1, numClasses) ;
  cPreds = p.cPreds ; bPreds = p.bPreds ; 

  for t = 1:numel(testIdx)
    % find predictions for current image
    cPreds_ = cPreds(:,:,t) ;  boxes = bPreds(:,:,t)' ; 
    keep = find(boxes(:,4) ~= 0) ; % drop unused RoIs
    boxes = boxes(keep,:) ; cPreds_ = cPreds_(:,keep) ; numKept = 0 ;

    for c = 1:numClasses - 1 % don't store bg 
      target = c + 1 ; % add offset for bg class
      % compute regressed proposals
      if ~opts.modelOpts.classAgnosticReg
        tBoxes = boxes(:,(target-1)*4+1:(target)*4) ;
      else
        tBoxes = boxes(:,5:8) ; % shared set of regressors
      end
      tScores = cPreds_(target,:)' ;
      cls_dets = [tBoxes tScores] ;
      
      % drop preds below threshold
      keep = find(cls_dets(:,end) >= opts.modelOpts.confThresh) ;
      cls_dets = cls_dets(keep,:) ;
      if ~numel(keep), continue ; end

      % TODO(samuel): Move last round of NMS into the computePredictions 
      % function for a fair timing benchmark (although this does match how
      % Girshick does it currently)
      % heuristic: keep a fixed number of dets per class per image before nms
      [~,si] = sort(cls_dets(:,5),'descend') ; cls_dets = cls_dets(si,:) ;
      numKeep = min(size(cls_dets,1),opts.modelOpts.maxPredsPerImage) ;
      cls_dets = cls_dets(1:numKeep,:) ;
      keep = vl_nnbboxnms(cls_dets', opts.modelOpts.nmsThresh) ;
      cls_dets = cls_dets(keep, :) ;

      if numel(keep)
        numKept = numKept + numel(keep) ;
        pBoxes = cls_dets(:,1:4) + 1 ; % Top left is (1,1) in VOC notation
        pScores = cls_dets(:,5) ;
        pBoxes = round(pBoxes, 2) ; % save storage space
        pScores = round(pScores, 5) ;
        switch opts.dataOpts.resultsFormat
          case 'minMax', % do nothing
          case 'minWH', pBoxes = [ pBoxes(:, 1:2) pBoxes(:,3:4) - pBoxes(:,1:2) ] ;
          otherwise, error('%s not recognised', opts.dataOpts.resultsFormat) ;
        end
        scores{c} = vertcat(scores{c}, pScores) ;
        bboxes{c} = vertcat(bboxes{c}, pBoxes) ;
        
        switch opts.dataOpts.name % ids are used differently by the datasets
          case 'pascal', pId = imdb.images.name{testIdx(t)} ;
          case 'coco', pId = imdb.images.id(testIdx(t)) ;
        end
        imageIds{c} = vertcat(imageIds{c}, repmat({pId}, size(pScores))) ; 
      end
    end

    if mod(t,100) == 1, fprintf('extracting %d/%d\n', t, numel(testIdx)) ; end
  end

  decodedPreds.imageIds = imageIds ;
  decodedPreds.scores = scores ;
  decodedPreds.bboxes = bboxes ;

% -------------------------------------------------------
function p = computePredictions(net, imdb, testIdx, opts) 
% -------------------------------------------------------
  prepareGPUs(opts, true) ;
  p = struct() ; params.testIdx = testIdx ;

  if numel(opts.gpus) <= 1
     state = processDetections(net, imdb, params, opts) ;
     p.cPreds = state.clsPreds ; p.bPreds = state.bboxPreds ;
  else
    topK = opts.modelOpts.maxPreds ; numClasses = opts.modelOpts.numClasses ;
    p.clsPreds = zeros(numClasses, topK, numel(testIdx), 'single') ; 
    if opts.modelOpts.classAgnosticReg, b = 8 ; else, b = 4*numClasses ; end
    p.bboxPreds = zeros(b, topK, numel(testIdx), 'single') ; 
    startup ;  % fix for parallel oddities
    spmd
     state = processDetections(net, imdb, params, opts) ;
    end
    for i = 1:numel(opts.gpus)
      state_ = state{i} ;
      p.cPreds(:,:,state_.computedIdx) = state_.clsPreds ;
      p.bPreds(:,:,state_.computedIdx) = state_.bboxPreds ;
    end
    p = rmfield(p, 'bboxPreds') ; p = rmfield(p, 'clsPreds') ; % clean up
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

  % The number of bbox predictions stored depends on whether the model makes
  % "per-class" predictions, or is agnostic to category for regression
  if opts.modelOpts.classAgnosticReg, b = 8 ; else, b = 4*numClasses ; end
  state.bboxPreds = zeros(b, topK, numel(computedIdx), 'single') ; 
  state.clsPreds = zeros(numClasses, topK, numel(computedIdx), 'single') ; 
  state.computedIdx = computedIdx ; 

  offset = 1 ; sc = sopts.scale ;

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
    if ~isempty(sc), args = {batch, opts, sc} ; else, args = {batch, opts} ; end
    inputs = opts.modelOpts.get_eval_batch(imdb, args{:}) ;

    if opts.batchOpts.prefetch
      batchStart_ = t + (labindex - 1) + opts.batchOpts.batchSize ;
      batchEnd_ = min(t + 2*opts.batchOpts.batchSize - 1, numel(testIdx)) ;
      next = testIdx(batchStart_: numlabs : batchEnd_) ;
      if ~isempty(sc), args = {next, opts, sc} ; else, args = {next, opts} ; end
      opts.modelOpts.get_eval_batch(imdb, args{:}, 'prefetch', true) ;
    end

    net.eval(inputs, 'test') ;
    storeIdx = offset:offset + numel(batch) - 1 ; 
    offset = offset + numel(batch) ;

    % THe final rounds of NMS will be done on the CPU during decoding
    cPreds = gather(net.getValue('cls_prob')) ; 
    bPreds = gather(net.getValue('bbox_pred')) ; 
    rois = gather(net.getValue('proposal')) ; 

    im_info = inputs{4} ; factor = im_info(3) ;
    imsz = round(im_info(1:2) / factor) ;
    boxes = (rois(2:end,:) - 1) / factor ; % undo offset required by roipool
    cBoxes = bboxTransformInv(boxes, squeeze(bPreds)) ;
    cBoxes = clipBoxes(cBoxes, imsz) ;

    state.clsPreds(:,1:size(cPreds,4),storeIdx) = cPreds ;
    state.bboxPreds(:,1:size(bPreds,4),storeIdx) = cBoxes ;
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
