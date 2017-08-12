function viz_get_batch(data, labels, targets)
% Visualize data 

  % batch size is always one with RPN
  labels = labels{1} ; targets = targets{1} ;
  showScores = 0 ; uClasses = unique(labels) ;
  % fix offset 
  %data = bsxfun(@plus, data, imMean) ;
  data = data + abs(min(data(:))) ; data = data / max(data(:)) ;
  classes = {'none_of_the_above', 'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', ...
    'train', 'tvmonitor'} ;

  for ii = 1:numel(uClasses)
    label = uClasses(ii) ; gt = targets(labels == label,:) ; clf ; disp('') ; 
    scores = ones(size(gt,1)) ; % arbitrary
    bboxDraw(gather(data), [gt scores], classes{label}, showScores) ;
  end

% ------------------------------------------------
function im = bboxDraw(im,boxes,label, showScores)
% ------------------------------------------------
% based on Ross Girshick's bbox drawing function
  imagesc(im) ; axis image ; axis off ; set(gcf, 'Color', 'white') ;
  lineOpts = {'color', 'r', 'linewidth', 2, 'linestyle', '-'} ;
  textOpts = {'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10} ;

  if ~isempty(boxes)
    x1 = boxes(:, 1) ; y1 = boxes(:, 2) ; 
    x2 = boxes(:, 3) ; y2 = boxes(:, 4) ;
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', lineOpts{:}) ;
    % uncomment to show confi
    if showScores
      for i = 1:size(boxes, 1)
        msg = [{sprintf('%.4f', boxes(i, end))} textOpts] ;
        text(double(x1(i)), double(y1(i)) - 2, msg{:}) ;
      end
    end
  end
  title(sprintf('gt boxes for %s', label)) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end
