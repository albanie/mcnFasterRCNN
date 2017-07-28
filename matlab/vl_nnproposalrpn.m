function y = vl_nnproposalrpn(x, b, imInfo, varargin)

  opts.fixed = [] ;
  opts.featStride = 16 ;
  opts.baseSize = 16 ;
  opts.minSize = 16 ;
  opts.scales = [8, 16, 32] ;
  opts.ratios = [0.5, 1, 2] ;
  opts.postNMSTopN = 300 ; % often 300 for test, 2000 for training
  opts.preNMSTopN = 6000 ; % often 6000 for test, 12000 for training
  opts.filterSmallProposals = true ;
  opts.nmsThresh = 0.7 ;
  opts.nms = 'gpu' ; % if mcnNMS has been compiled, NMS can be run on the GPU
  [opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  if ~isempty(opts.fixed), y = opts.fixed ; return ; end
  if ~isempty(dzdy), assert('this layer is a one way street') ; end

  anchors = generateAnchors(opts) ;
  numAnchors = size(anchors, 1) ;
  layerWidth = size(x, 2) ; layerHeight = size(x, 1) ;
  scores = x(:,:,numAnchors+1:end) ;
  shiftX = (0:layerWidth-1) .* double(opts.featStride) ;
  shiftY = (0:layerHeight-1) .* double(opts.featStride) ;
  [shiftX, shiftY] = meshgrid(shiftX, shiftY) ;
  shiftX_T = shiftX' ; shiftY_T = shiftY' ;
  shifts = [shiftX_T(:)  shiftY_T(:) shiftX_T(:)  shiftY_T(:) ] ;
  anchors = bsxfun(@plus, permute(anchors, [3 1 2]), reshape(shifts, [], 1, 4)) ;
  anchors = reshape(permute(anchors, [2 1 3]), [], 4) ;
  bboxDeltas = reshape(permute(b, [3 2 1]), 4, [])' ;
  scores = reshape(permute(scores, [3 2 1]), [], 1) ;
  proposals = bboxTransformInv(anchors, bboxDeltas) ;
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

  if opts.preNMSTopN, sIdx = sIdx(1:min(numel(sIdx), opts.preNMSTopN)) ; end
  proposals = proposals(sIdx,:) ; scores = scores(sIdx) ;

  switch opts.nms
    case 'gpu', keep = vl_nnbboxnms([proposals'; scores'], opts.nmsThresh) ;
    case 'cpu', keep = bbox_nms(gather([proposals scores]), opts.nmsThresh) ;
    otherwise, error('nms processingtype %s not recognised', opts.nms) ;
  end

  if opts.postNMSTopN, keep = keep(1:min(numel(keep), opts.postNMSTopN)) ; end

  proposals = proposals(keep,:) + 1 ; % fix indexing expected by ROI layer
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
