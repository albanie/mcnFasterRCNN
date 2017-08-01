function anchors = generateAnchors(opts)
%GENERATEANCHORS - generate anchor windows for given options
%  GENERATEANCHORS(OPTS) will generate a set of anchors by 
%  applying each of the scales and aspect ratios defined in 
%  OPTS.SCALES and OPTS.RATIOS to o "base anchor", which is 
%  a square of witdh OPTS.BASESIZE. 

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
