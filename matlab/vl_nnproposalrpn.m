function y = vl_nnproposalrpn(x, b, imInfo, varargin)

opts.fixed = [] ;
opts.featStride = 16 ;
opts.baseSize = 16 ;
opts.minSize = 16 ;
opts.scales = [8, 16, 32] ;
opts.ratios = [0.5, 1, 2] ;
opts.postNMSTopN = 300 ;
opts.preNMSTopN = 6000 ;
opts.filterSmallProposals = true ;
opts.nmsThresh = 0.7 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if ~isempty(opts.fixed)
  y = opts.fixed ; return ;
end

anchors = generateAnchors(opts) ;
numAnchors = size(anchors, 1) ;

% Each spatial element of the input layer `im` produces a corresponding
% prior box in the input image. We assume that every image in the
% batch is the same size so that the prior boxes can be duplicated
% across all input images. 
layerWidth = size(x, 2) ; layerHeight = size(x, 1) ;
imgWidth = imInfo(2) ; imgHeight = imInfo(1) ;

scores = x(:,:,numAnchors+1:end) ;

shiftX = [0:layerWidth-1] .* double(opts.featStride) ;
shiftY = [0:layerHeight-1] .* double(opts.featStride) ;
[shiftX, shiftY] = meshgrid(shiftX, shiftY) ;
shiftX_T = shiftX' ; shiftY_T = shiftY' ;
shifts = [shiftX_T(:)  shiftY_T(:) shiftX_T(:)  shiftY_T(:) ] ;
shifts2 = reshape(shifts, [], 1, 4) ;
anchors2 = permute(anchors, [3 1 2]) ;
anchors = bsxfun(@plus, anchors2, shifts2) ;
anchors3 = reshape(permute(anchors, [2 1 3]), [], 4) ;
anchors = anchors3 ;

bboxDeltas = reshape(permute(b, [3 2 1]), 4, [])' ;
scores = reshape(permute(scores, [3 2 1]), [], 1) ;

proposals = bboxTransformInv(anchors, bboxDeltas) ;

% clip proposals
proposals = clipProposals(proposals, imInfo) ;

if opts.filterSmallProposals 
  % An observation was made in the following paper that this filtering
  % may not be helpful:
  % An Implementation of Faster RCNN with Study for Region Sampling, 
  % X. Chen, A. Gupta, https://arxiv.org/pdf/1702.02138.pdf, 2017
  keep = filterPropsoals(proposals, opts.minSize *imInfo(3)) ;
  proposals = proposals(keep,:) ;
  scores = scores(keep) ;
end

[~,sIdx] = sort(scores, 'descend') ;

if opts.preNMSTopN > 0, sIdx = sIdx(1:min(numel(sIdx), opts.preNMSTopN)) ; end
proposals = proposals(sIdx,:) ;
scores = scores(sIdx) ;

cpuData = gather([proposals scores]) ; % faster on CPU
keep = bbox_nms(cpuData, opts.nmsThresh) ;

if opts.postNMSTopN > 0, keep = keep(1:min(numel(keep), opts.postNMSTopN)) ; end
proposals = proposals(keep,:) + 1 ; % fix indexing
scores = scores(keep) ;
imIds = ones(1, numel(keep)) ;
y = vertcat(imIds, proposals') ;

% ---------------------------------------------------
function keep = filterPropsoals(proposals, minSize)
% ---------------------------------------------------
  W = proposals(:,3) - proposals(:,1) + 1 ;
  H = proposals(:,4) - proposals(:,2) + 1 ;
  keep = find(W >= minSize & H >= minSize) ;

% ---------------------------------------------------
function proposals = clipProposals(proposals, imInfo)
% ---------------------------------------------------
  proposals(:, 1) =  max(min(proposals(:, 1), imInfo(2) -1), 0) ;
  proposals(:, 2) =  max(min(proposals(:, 2), imInfo(1) -1), 0) ;
  proposals(:, 3) =  max(min(proposals(:, 3), imInfo(2) -1), 0) ;
  proposals(:, 4) =  max(min(proposals(:, 4), imInfo(1) -1), 0) ;

% --------------------------------------------------------
function proposals = bboxTransformInv(anchors, bboxDeltas) 
% --------------------------------------------------------
  widths = anchors(:,3) - anchors(:,1) + 1 ;
  heights = anchors(:,4) - anchors(:,2) + 1 ;
  ctrX = anchors(:,1) + 0.5 .* widths ;
  ctrY = anchors(:,2) + 0.5 .* heights ;

  dx = bboxDeltas(:,1) ;
  dy = bboxDeltas(:,2) ;
  dw = bboxDeltas(:,3) ;
  dh = bboxDeltas(:,4) ;

  predCenX = dx .* widths + ctrX ;
  predCenY = dy .* heights + ctrY ;
  predW = exp(dw) .* widths ; 
  predH = exp(dh) .* heights ;
  proposals = [ predCenX - 0.5 * predW ...
            predCenY - 0.5 * predH ...
            predCenX + 0.5 * predW ...
            predCenY + 0.5 * predH] ;

% --------------------------------------
function anchors = generateAnchors(opts)
% --------------------------------------
  baseAnchor = [1, 1, opts.baseSize, opts.baseSize] - 1;
  baseCenWH = anchorCoder(baseAnchor, 'CenWH') ;
  sz = prod(baseCenWH([3 4])) ;
  sizeRatios = sz ./ opts.ratios ;
  W = round(sqrt(sizeRatios)) ; H = round(W .* opts.ratios) ;
  cen = repmat(baseCenWH(1:2), [numel(W) 1]) ;
  boxes = [ cen W' H'] ;
  ratioAnchors = anchorCoder(boxes, 'anchor') ;

  % compute scale anchors
  m = numel(opts.ratios) ;  n = numel(opts.scales) ;
  scaleAnchors = arrayfun(@(i) {repmat(ratioAnchors(i,:), n, 1)}, 1:m) ;
  scaleAnchors = vertcat(scaleAnchors{:}) ;
  scaleBoxes = anchorCoder(scaleAnchors, 'CenWH') ;
  z = repmat(opts.scales', [numel(opts.ratios) 2]) ; 
  scaleBoxes(:,3:4) = scaleBoxes(:,3:4) .* z ;
  anchors = anchorCoder(scaleBoxes, 'anchor') ;
