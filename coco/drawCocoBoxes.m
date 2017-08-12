function drawCocoBoxes(im, box, confidence, label, varargin) 
%DRAWCOCOBOXES - draw bounding boxes predicted by coco detector
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.format = 'MinMax' ;
  opts.gt = [] ;
  opts = vl_argparse(opts, varargin) ;

  figure ; im = im / 255 ;
  switch opts.format
    case 'MinWH', rectangle = [ box(1:2) box(3:4)] ;
    case 'MinMax', rectangle = [ box(1:2) box(3:4) - box(1:2)] ;
    otherwise, error('box format not recognised') 
  end
  im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 3, 'Color', 'red');
  if ~isempty(opts.gt) % add gt if provided
    [revLabelMap,labels] = getCocoLabelMap('reverse', 1) ;
    for ii = 1:numel(opts.gt)
      gt_ = opts.gt(ii) ; rectangle = gt_.bbox ; 
      gtLabel = labels{revLabelMap(gt_.category_id)} ;
      im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 3, 'Color', 'g') ;
      fprintf('adding gt box with label %s\n', gtLabel) ;
    end
  end

  imagesc(im) ;
  title(sprintf('top Faster RCNN pred: %s\n conf: %f', label, confidence)) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end
