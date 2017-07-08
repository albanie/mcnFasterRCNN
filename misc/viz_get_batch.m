function viz_get_batch(data, labels, targets)
% Visualize data 
% batch size is always one with RPN
labels = labels{1} ; targets = targets{1} ;

sz = size(data) ; uClasses = unique(labels) ;
classes = {'none_of_the_above', 'aeroplane', 'bicycle', 'bird', ...
  'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', ...
  'train', 'tvmonitor'} ;

for ii = 1:numel(uClasses)
  label = uClasses(ii) ; gt = targets(labels == label,:) ;
  boxes = gt .* sz([2 1 2 1]) ; scores = ones(size(gt,1)) ; % arbitrary
  clf ; disp('') ; bbox_draw(gather(data)/255, [boxes scores]) ;
  title(sprintf('gt boxes for %s', classes{label})) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end
end
