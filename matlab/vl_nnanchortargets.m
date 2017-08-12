function [l, t, iw, ow, cw] = vl_nnanchortargets(x, gb, imInfo, varargin)
%VL_NNANCHORTARGETS produces training targets for RPN
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.baseSize = 16 ;
  opts.featStride = 16 ;
  opts.rpnFGRatio = 0.5 ;
  opts.insideWeight = 1 ;
  opts.rpnBatchSize = 256 ;
  opts.scales = [8, 16, 32] ;
  opts.ratios = [0.5, 1, 2] ;
  opts.rpnPositiveWeight = -1 ;
  opts.rpnNegativeOverlap = 0.3 ;
  opts.rpnPositiveOverlap = 0.7 ;
  opts.allowedBorder = false ;
  opts.clobberPositives = false ;
  opts.filterSmallProposals = true ;
  opts.negLabel = 1 ; % note: this is different to caffe (-1=ignore, 0=neg)
  opts.posLabel = 2 ;  
  opts.ignoreLabel = 0 ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  layerHeight = size(x, 1) ;layerWidth = size(x, 2) ; 
  anchors = generateAnchors(opts) ; numAnchors = size(anchors, 1) ;
  batchSize = size(x,4) ; assert(batchSize == 1, 'only batch size 1 support') ;

  % labels, targets and  'inside'/'outside' weights
  l = zeros(layerHeight, layerWidth*numAnchors, 1, batchSize, 'like', x) ;
  t = zeros(layerHeight, layerWidth, numAnchors*4, batchSize, 'like', x) ;
  iw = zeros(layerHeight, layerWidth, numAnchors*4, batchSize, 'like', x) ;
  ow = zeros(layerHeight, layerWidth, numAnchors*4, batchSize, 'like', x) ;

  for bb = 1:batchSize
    gtBoxes = gb{bb} ;

    % compute proposals using anchors and bbox deltas
    shiftX = (0:layerWidth-1) .* double(opts.featStride) ;
    shiftY = (0:layerHeight-1) .* double(opts.featStride) ;
    [shiftX, shiftY] = meshgrid(shiftX, shiftY) ;
    shiftX_T = shiftX' ; shiftY_T = shiftY' ;
    shifts = [shiftX_T(:)  shiftY_T(:) shiftX_T(:)  shiftY_T(:) ] ;
    reshaped = reshape(shifts, [], 1, 4) ;
    anchors = bsxfun(@plus, permute(anchors, [3 1 2]), reshaped) ;
    allAnchors = reshape(permute(anchors, [2 1 3]), [], 4) ;
    totalAnchors = size(allAnchors, 1) ;

    % restrict to anchors that lie inside the image
    idxInside = find((allAnchors(:,1) >= -opts.allowedBorder) & ...
           (allAnchors(:,2) >= -opts.allowedBorder) & ...
           (allAnchors(:,3) < imInfo(2) + opts.allowedBorder) & ...
           (allAnchors(:,4) < imInfo(1) + opts.allowedBorder)) ;
    anchors = allAnchors(idxInside,:) ;
    labels = ones(numel(idxInside),1) * opts.ignoreLabel ; % 1 +ve, 0 -ve, -1 ignore

    % compute overlaps
    overlaps = bbox_overlap(gtBoxes, anchors) ;
    [maxOverlaps, mI] = max(overlaps, [], 1) ; % dim must be written explicitly
    [gtMaxOverlaps, ~] = max(overlaps, [], 2) ;
    [~,~,ib] = intersect(gtMaxOverlaps, overlaps, 'stable') ; 
    [~,gtI] = ind2sub(size(overlaps),ib) ; % approx 50% faster than using find()

    if ~opts.clobberPositives 
      labels(maxOverlaps < opts.rpnNegativeOverlap) = opts.negLabel ;
    end

    % for each gt, assign anchor with highest overlap and all above IoU thresh
    labels(gtI) = opts.posLabel ; 
    labels(maxOverlaps >= opts.rpnPositiveOverlap) = opts.posLabel ;

    if opts.clobberPositives 
      labels(maxOverlaps < opts.rpnNegativeOverlap) = opts.negLabel ;
    end

    % subsample positive labels if required
    numPos = floor(opts.rpnFGRatio * opts.rpnBatchSize) ;
    fgInds = find(labels == opts.posLabel) ; excess = numel(fgInds) - numPos ;
    if excess > 0 
      dropIdx = fgInds(randsample(numel(fgInds), excess)) ;
      labels(dropIdx) = opts.ignoreLabel ;
    end

    % subsample negative labels if required
    numNeg = opts.rpnBatchSize - sum(labels == opts.posLabel) ;
    bgInds = find(labels == opts.negLabel) ; excess = numel(bgInds) - numNeg ;
    if excess > 0 
      dropIdx = bgInds(randsample(numel(bgInds), excess)) ;
      % temp fix to numerically reproduce python code
      %tmp = load('drops.mat') ; dropIdx = tmp.drops + 1 ;
      labels(dropIdx) = opts.ignoreLabel ;
    end

    bboxTargets = bbox_transform(anchors, gtBoxes(mI,:)) ;
    bboxInsideWeights = zeros(numel(idxInside), 4) ;
    bboxOutsideWeights = zeros(numel(idxInside), 4) ;
    bboxInsideWeights(labels == opts.posLabel,:) = opts.insideWeight ;

    if opts.rpnPositiveWeight < 0 % uniform weighting of samples
      numExamples = sum(labels ~= opts.ignoreLabel) ;
      posWeights = ones(1,4) .* (1 / numExamples) ;
      negWeights = ones(1,4) .* (1 / numExamples) ;
    else
      msg = 'RPN positive weight must lie in [0,1]' ;
      assert(opts.rpnPositiveWeight > 0 & opts.rpnPositiveWeight < 1, msg) ;
      posWeights = opts.rpnPositiveWeight / sum(labels == opts.posLabel) ;
      negWeights = (1 - opts.rpnPositiveWeight) / sum(labels == opts.negLabel) ;
    end

    pos = find(labels == opts.posLabel) ; neg = find(labels == opts.negLabel) ;
    bboxOutsideWeights(pos,:) = repmat(posWeights, numel(pos), 1) ;
    bboxOutsideWeights(neg,:) = repmat(negWeights, numel(neg), 1) ;

    labels = unMap(labels, totalAnchors, idxInside, opts.ignoreLabel) ;
    bboxTargets = unMap(bboxTargets, totalAnchors, idxInside, 0) ;
    bboxInsideWeights = unMap(bboxInsideWeights, totalAnchors, idxInside, 0) ;
    bboxOutsideWeights = unMap(bboxOutsideWeights, totalAnchors, idxInside, 0) ;

    % prepare for reshape fu to align the zodiac
    W = layerWidth ; H = layerHeight ; A = numAnchors ;

    % To be compatible with the expected loss function shapes and keep things
    % simple, the labels are reshape to have shape H x (W*A) x 1 x N (where N is
    % the batch size (always 1).  
    l(:,:,:,bb) = reshape(permute(reshape(labels, A, W, H), [3 2 1]), H, W*A) ;
    t(:,:,:,bb) = permute(reshape(bboxTargets', A*4, W, H), [3 2 1]) ;
    iw(:,:,:,bb) = permute(reshape(bboxInsideWeights', A*4, W, H), [3 2 1]) ;
    ow(:,:,:,bb) = permute(reshape(bboxOutsideWeights', A*4, W, H), [3 2 1]) ;
    cw = 1 / sum(labels ~= opts.ignoreLabel) ; % instance weights for classifier
  end

%  ------------------------------------------
function res = unMap(vals, count, idx, fill)
%  ------------------------------------------
  res = ones(count, size(vals,2)) * fill ; 
  res(idx,:) = vals ; 
  
