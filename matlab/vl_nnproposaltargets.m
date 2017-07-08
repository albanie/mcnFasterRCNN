function [r, l, t, iw, ow] = vl_nnproposaltargets(p, gb, gl, varargin)
%VL_NNANCHORTARGETS produces training targets for RPN

  opts.baseSize = 16 ;
  opts.numClasses = 21 ;
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
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  keyboard

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
    anchors = bsxfun(@plus, permute(anchors, [3 1 2]), ...
                                        reshape(shifts, [], 1, 4)) ;
    allAnchors = reshape(permute(anchors, [2 1 3]), [], 4) ;
    totalAnchors = size(allAnchors, 1) ;

    % restrict to anchors that lie inside the image
    idxInside = find((allAnchors(:,1) >= -opts.allowedBorder) & ...
           (allAnchors(:,2) >= -opts.allowedBorder) & ...
           (allAnchors(:,3) < imInfo(2) + opts.allowedBorder) & ...
           (allAnchors(:,4) < imInfo(1) + opts.allowedBorder)) ;
    anchors = allAnchors(idxInside,:) ;
    labels = zeros(numel(idxInside),1) * -1 ; % 1 +ve, 0 -ve, -1 ignore

    % compute overlaps
    overlaps = bbox_overlap(gtBoxes, anchors) ;
    [maxOverlaps, mI] = max(overlaps) ;
    [gtMaxOverlaps, ~] = max(overlaps, [], 2) ;
    [~,gtI] = find(overlaps == gtMaxOverlaps) ;

    if ~opts.clobberPositives 
      labels(maxOverlaps < opts.rpnNegativeOverlap) = 0 ;
    end

    % for each gt, assign anchor with highest overlap and all above IoU thresh
    labels(gtI) = 1 ; labels(maxOverlaps >= opts.rpnPositiveOverlap) = 1 ;

    if opts.clobberPositives 
      labels(maxOverlaps < opts.rpnNegativeOverlap) = 0 ;
    end

    % subsample positive labels if required
    numPos = floor(opts.rpnFGRatio * opts.rpnBatchSize) ;
    fgInds = find(labels == 1) ; excess = numel(fgInds) - numPos ;
    if excess > 0 
      dropIdx = fgInds(randsample(numel(fgInds), excess)) ;
      labels(dropIdx) = -1 ;
    end

    % subsample negative labels if required
    numNeg = opts.rpnBatchSize - sum(labels == 1) ;
    bgInds = find(labels == 0) ; excess = numel(bgInds) - numNeg ;
    if excess > 0 
      %dropIdx = bgInds(randsample(numel(bgInds), excess)) ;
      % temp fix to numerically reproduce python code
      tmp = load('drops.mat') ;
      dropIdx = tmp.idx + 1 ;
      labels(dropIdx) = -1 ;
    end

    bboxTargets = bbox_transform(anchors, gtBoxes(mI,:)) ;
    bboxInsideWeights = zeros(numel(idxInside), 4) ;
    bboxOutsideWeights = zeros(numel(idxInside), 4) ;
    bboxInsideWeights(labels == 1,:) = opts.insideWeight ;

    if opts.rpnPositiveWeight < 0 % uniform weighting of samples
      numExamples = sum(labels >= 0) ;
      posWeights = ones(1,4) .* (1 / numExamples) ;
      negWeights = ones(1,4) .* (1 / numExamples) ;
    else
      msg = 'RPN positive weight must lie in [0,1]' ;
      assert(opts.rpnPositiveWeight > 0 & opts.rpnPositiveWeight < 1, msg) ;
      posWeights = opts.rpnPositiveWeight / sum(labels == 1) ;
      negWeights = (1 - opts.rpnPositiveWeight) / sum(labels == 0) ;
    end

    pos = find(labels == 1) ; neg = find(labels == 0) ;
    bboxOutsideWeights(pos,:) = repmat(posWeights, numel(pos), 1) ;
    bboxOutsideWeights(neg,:) = repmat(negWeights, numel(neg), 1) ;

    labels = unMap(labels, totalAnchors, idxInside, -1) ;
    bboxTargets = unMap(bboxTargets, totalAnchors, idxInside, 0) ;
    bboxInsideWeights = unMap(bboxInsideWeights, totalAnchors, idxInside, 0) ;
    bboxOutsideWeights = unMap(bboxOutsideWeights, totalAnchors, idxInside, 0) ;

    % prepare for reshape fu to align the zodiac
    W = layerWidth ; H = layerHeight ; A = numAnchors ;

    % To be compatible with the expected loss function shapes and keep things
    % simple, the labels are reshape to have shape H x (W*A) x 1 x N (where N is
    % the batch size (always 1).  

    l(:,:,:,bb) = reshape(permute(reshape(labels, A, W, H), [3 2 1]), H, W*A) ;
    %l(:,:,:,bb) = reshape(permute(reshape(labels, A, W, H), [3 2 1]), 1, 1, H*W*A) ;
    t(:,:,:,bb) = permute(reshape(bboxTargets', A*4, W, H), [3 2 1]) ;
    iw(:,:,:,bb) = permute(reshape(bboxInsideWeights', A*4, W, H), [3 2 1]) ;
    ow(:,:,:,bb) = permute(reshape(bboxOutsideWeights', A*4, W, H), [3 2 1]) ;
  end

%  ------------------------------------------
function res = unMap(vals, count, idx, fill)
%  ------------------------------------------
  res = ones(count, size(vals,2)) * fill ; 
  res(idx,:) = vals ; 
  
