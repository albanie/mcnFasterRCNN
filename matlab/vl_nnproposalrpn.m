function y = vl_nnproposalrpn(x, b, imInfo, varargin)
%VL_NNPROPOSALRPN generate a set of region proposals 
%  VL_NNPROPOSALRPN(X, B, IMINFO) generates a set of region proposals
%  from a set of HxWx(2*A)xN objectness scores X, and H x W x bounding box 
%  regression predictions B with shape HxWx(4*A)xN, where N is the batch
%  size, A is the number of anchors at each spatial location and H and W are
%  the height and width of the feature map used to source the predictions.
%  
%  The total number of anchors at each location is a product of the number of
%  scales and the number of aspect ratios used at each location (for example,
%  in the original pascal VOC model, 9 (=3x3) were used.
% 
%  The proposal generation algorithm introduced by faster R-CNN is as follows:
%
%    For each spatial location (h,w) where 1 <= h <= H, 1 <= w <= W,
%      apply the set of bbox deltas b(h,w,:) to each of the A anchors
%    constrain predicted boxes to lie wihtin image boundary by clipping
%    remove boxes that are too narrow or too short (below a pixel threshold)
%    rank proposals (i.e. (box, score) pairs) in descending order
%    drop all except the top preNMSTopN proposals
%    apply NMS with threshold `nmsThresh` to remaining proposals
%    return the top postNMSTopN proposals

  opts.fixed = [] ;
  opts.minSize = 16 ;
  opts.baseSize = 16 ;
  opts.featStride = 16 ;
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


  anchors = generateAnchors(opts) ; % generate a fixed set of anchors
  numAnchors = size(anchors, 1) ;
  layerWidth = size(x, 2) ; layerHeight = size(x, 1) ;

  % create an evenly spaced grid and attach anchors to each location
  scores = x(:,:,numAnchors+1:end) ;
  shiftX = (0:layerWidth-1) .* double(opts.featStride) ;
  shiftY = (0:layerHeight-1) .* double(opts.featStride) ;
  [shiftX, shiftY] = meshgrid(shiftX, shiftY) ;
  shiftX_T = shiftX' ; shiftY_T = shiftY' ;
  shifts = [shiftX_T(:)  shiftY_T(:) shiftX_T(:)  shiftY_T(:) ] ;
  % restructured to improve speed
  anchors = bsxfun(@plus, permute(anchors, [1 3 2]), reshape(shifts, 1, [], 4)) ;

  % reshape to a common layout to apply bbox predictions
  anchors = reshape(anchors, [], 4)' ;
  bboxDeltas = reshape(permute(b, [3 2 1]), 4, []) ;
  scores = reshape(permute(scores, [3 2 1]), 1, []) ;
  proposals = bboxTransformInv(anchors, bboxDeltas) ;
  proposals = clipBoxes(proposals, imInfo) ;

  if opts.filterSmallProposals 
    % An observation was made in the following paper that this filtering
    % may not be helpful:
    % An Implementation of Faster RCNN with Study for Region Sampling, 
    % X. Chen, A. Gupta, https://arxiv.org/pdf/1702.02138.pdf, 2017
    keep = filterPropsoals(proposals, opts.minSize *imInfo(3)) ;
    proposals = proposals(:,keep) ;
    scores = scores(keep) ;
  end
  [~,sIdx] = sort(scores, 'descend') ;

  % apply pre-NMS proposal cutoff
  if opts.preNMSTopN, sIdx = sIdx(1:min(numel(sIdx), opts.preNMSTopN)) ; end
  proposals = proposals(:,sIdx) ; scores = scores(sIdx) ;

  switch opts.nms
    case 'gpu', keep = vl_nnbboxnms([proposals ; scores], opts.nmsThresh) ;
    case 'cpu', keep = bbox_nms(gather([proposals' scores']), opts.nmsThresh) ;
    otherwise, error('nms processing type %s not recognised', opts.nms) ;
  end

  % apply post-NMS proposal cutoff
  if opts.postNMSTopN, keep = keep(1:min(numel(keep), opts.postNMSTopN)) ; end
  proposals = proposals(:,keep) + 1 ; % fix indexing expected by ROI layer

  imIds = ones(1, numel(keep)) ; 
  y = vertcat(imIds, proposals) ;

% -------------------------------------------------
function keep = filterPropsoals(proposals, minSize)
% -------------------------------------------------
  W = proposals(3,:) - proposals(1,:) + 1 ;
  H = proposals(4,:) - proposals(2,:) + 1 ;
  keep = find(W >= minSize & H >= minSize) ;
