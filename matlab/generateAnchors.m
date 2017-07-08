function anchors = generateAnchors(opts)

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
