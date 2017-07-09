function [r, l, t, iw, ow, cw] = vl_nnproposaltargets(p, gb, gl, varargin)
%VL_NNPROPOSALTARGETS produces training targets for proposals

  opts.roiBatchSize = 128 ;
  opts.bgClass = 1 ;
  opts.numClasses = 21 ;
  opts.fgRatio = 0.5 ;
  opts.fgThresh = 0.5 ;
  opts.bgThreshLo = 0 ;
  opts.bgThreshHi = 0.5 ;
  opts.negLabel = 1 ; % note: this is different to caffe (-1=ignore, 0=neg)
  opts.insideWeight = 1 ;
  opts.normalizeTargets = 1 ;
  opts.normalizeMeans = [0, 0, 0, 0] ;
  opts.normalizeStdDevs = [0.1, 0.1, 0.2, 0.2] ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  %p = p - 1 ; % TODO(sam) cleanup indexing
  batchSize = numel(gb) ; assert(batchSize == 1, 'only batch size 1 support') ;
  gtBoxes = gb{1} ; gtLabels = gl{1} ;

  % add ground truth boxes to candidates
  p = [ p' ; [ones(size(gtBoxes,1),1) gtBoxes] ] ;
  roisPerImage = opts.roiBatchSize / batchSize ;
  [l, r, t, iw] = sampleRois(p, gtBoxes, gtLabels, roisPerImage, opts) ;
  ow = zeros(size(iw)) ; 
  ow(iw > 0) = 1 ./ numel(l) ; % handle batch size normalisation here
  cw = 1 / numel(l) ; % instance weights for classifier

% ---------------------------------------------------------------------
function [labels, rois, targets, iw] = sampleRois(allRois, gtBoxes, ...
                                             gtLabels, roisPerImage, opts)
% ---------------------------------------------------------------------
% sample a set of RoIs that achieve the desired balance between foreground
% and background

fgRoisPerImage = round(opts.fgRatio * roisPerImage) ;
overlaps = bbox_overlap(gather(allRois(:,2:end)), gtBoxes) ; % run on CPU
[maxOverlaps, gtI] = max(overlaps, [], 2) ;
labels = reshape(gtLabels(gtI), 1, []) ; % ensure 1xn shape for single labels

fgInds = find(maxOverlaps >= opts.fgThresh) ;
% prevent issue caused by fgRoisPerImage < numel(fgInds)
numFgRois = min(fgRoisPerImage, numel(fgInds)) ;

if numel(fgInds) > 0
  fgInds = fgInds(randsample(numel(fgInds), numFgRois)) ;
  % temp fix to numerically reproduce python code
  %tmp = load('fg_inds.mat') ; 
  %fgInds = tmp.inds + 1 ;
end

% pick bg rois that lie inside the threshold interval
bgInds = find(maxOverlaps < opts.bgThreshHi & maxOverlaps >= opts.bgThreshLo) ;
numBgRois = roisPerImage - numFgRois ;
numBgRois = min(numBgRois, numel(bgInds)) ;

if numel(bgInds) > 0
  bgInds = bgInds(randsample(numel(bgInds), numBgRois)) ;
  % temp fix to numerically reproduce python code
  %tmp = load('bg_inds.mat') ; 
  %bgInds = tmp.inds + 1 ;
end

keep = [fgInds ; bgInds] ;
labels = labels(keep) ;
labels(numFgRois+1:end) = opts.negLabel ; % set labels to bg
rois = allRois(keep,:) ;
targetData = computeTargets(rois(:,2:end), gtBoxes(gtI(keep),:), labels, opts) ;
[targets, iw] = getBboxRegressionLabels(targetData, opts) ;
rois = rois' ; % mcn expected roi shape

% ------------------------------------------------------------
function targetData = computeTargets(rois, gtBoxes, labels, opts) 
% ------------------------------------------------------------
  targets = bbox_transform(rois, gtBoxes) ;
  if opts.normalizeTargets % normalize targets by mean and std dev
    centered = bsxfun(@minus, targets, opts.normalizeMeans) ;
    targets = bsxfun(@rdivide, centered, opts.normalizeStdDevs) ;  
  end
  targetData = [labels' targets] ;

% --------------------------------------------------------------
function [targets,iw] = getBboxRegressionLabels(targetData, opts) 
% ---------------------------------------------------------------
% GETBBOXREGRESSIONLABELS transforms the given targetData into the form
% expected by the loss function
%
% `targetData` is a N x 5 array of the form (class, tx, ty, tw, th)
% 
% Returns an 1 1 x (4*numClasses) x N array of regression targets (where only
% the ground truth class takes non-zero target values
  classes = targetData(:,1) ;
  targets = zeros(1,1,4 * opts.numClasses, numel(classes), 'like', targetData) ;
  iw = zeros(size(targets)) ;
  inds = find(classes ~= opts.bgClass) ;
  for ii = 1:numel(inds)
    ind = inds(ii) ;
    cls = classes(ind) ;
    head = 4 * (cls-1) ; tail = head + 4 ;
    targets(:,:,head+1:tail, ind) = targetData(ind, 2:end) ;
    iw(:,:,head+1:tail, ind) = opts.insideWeight ;
  end
