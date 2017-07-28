function drawCocoBoxes(im, box, confidence, label, varargin) 

opts.format = 'MinMax' ;
opts.gt = [] ;
opts = vl_argparse(opts, varargin) ;

figure ; im = im / 255 ;
switch opts.format
  case 'MinMax'
    rectangle = [ box(1:2) box(3:4) - box(1:2)] ;
  case 'MinWH'
    rectangle = [ box(1:2) box(3:4)] ;
  otherwise
    error('box format not recognised') 
end
im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 3, 'Color', 'red');

% add gt if provided
if ~isempty(opts.gt)
  [revLabelMap,labels] = getCocoLabelMap('reverse', 1) ;
  for ii = 1:numel(opts.gt)
    gt_ = opts.gt(ii) ; rectangle = gt_.bbox ; 
    gtLabel = labels{revLabelMap(gt_.category_id)} ;
    im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 3, 'Color', 'g') ;
    fprintf('adding gt box with label %s\n', gtLabel) ;
  end
end

imagesc(im) ;
title(sprintf('top Faster RCNN prediction: %s\n conf: %f', label, confidence)) ;
if exist('zs_dispFig'), zs_dispFig ; end
