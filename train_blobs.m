full = Layer.fromCompiledNet(net) ;
%tail_cls = full{1}.find('rpn_loss_cls',1) ;
%tail_bbox = full{1}.find('rpn_loss_bbox',1) ;
tail_loss = full{1}.find('rpn_multitask_loss', 1) ;

% squeeze bias
%tail.find('conv1_1',1).inputs{3}.value = squeeze(tail.find('conv1_1',1).inputs{3}.value) ;

%tail.find('conv1_1',1).inputs{2}.value = permute(tail.find('conv1_1',1).inputs{2}.value, [ 1 2 3 4]) ;

%f = tail.find('conv1_1',1).inputs{2}.value ;
%b = tail.find('conv1_1',1).inputs{3}.value ;

% set bias to zero
%tail.find('conv1_1',1).inputs{3}.value = zeros(size(tail.find('conv1_1',1).inputs{3}.value), 'single') ;

tailNet = Net(tail_loss) ;

im = permute(store.data, [3 4 2 1]) ; im = im(:,:, [3 2 1]) ;
l = {store.gt_boxes(5) + 1} ; 
b = {store.gt_boxes(1:4)} ; 
imInfo = store.im_info ;
batchData = {'data', im, 'gtBoxes', b, 'imInfo', imInfo} ;
%batchData = {'data', im} ;
tailNet.eval(batchData, 'forward') ;

% check feats
%info = tailNet.getVarInfo() ;
%keep = find(cellfun(@(x) strcmp(x, 'layer'), {info.type})) ;
sample = {'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'conv5_3', ...
          'rpn_cls_score', 'rpn_bbox_pred', ...
          'rpn_cls_score_reshape', 'rpn_labels', 'rpn_bbox_targets', ...
          'rpn_bbox_inside_weights', 'rpn_bbox_outside_weights', ...
          'rpn_loss_bbox', 'rpn_loss_cls'} ;


offset = 0 ;
for ii = 1:numel(sample)
  vName = sample{ii} ; vName_ = vName ;
  tokens = regexp(vName, 'relu(\d[_]\d)', 'tokens') ;
  if ~isempty(tokens), vName_ = sprintf('conv%s', tokens{1}{1}) ; end
  if strcmp(vName, 'rpn_bbox_loss'), vName_ = 'rpn_loss_bbox' ; end

  switch vName
    case 'rpn_labels' 
      offset = 1 ;
    case 'rpn_bbox_targets' 
      offset = 2 ; vName = 'rpn_labels' ;
    case 'rpn_bbox_inside_weights' 
      offset = 3 ; vName = 'rpn_labels' ;
    case 'rpn_bbox_outside_weights' 
      offset = 3 ; vName = 'rpn_labels' ;
  end

  x = tailNet.getValue(vName) ; 
  x_ = permute(store.(vName_), [3 4 2 1]) ;
  if strcmp(vName, 'rpn_labels'), x = x{offset} ; end

  %if strcmp(vName, 'rpn_cls_score_reshape'), keyboard ; end
  %if strcmp(vName, 'rpn_loss_cls'), keyboard ; end

  if offset == 1 
    % to account for column major ordering, we are using a different 
    % label shape i.e. Hx(WxA)x1x1 vs 1x1x(AxH)xW where H is faster than A in caffe
    numAnchors = 9 ;

    % match caffe convention of -1 ignore, 0 negative
    x(x == -1) = -2 ; x(x == 0) = -1 ; x(x == -2) = 0 ;

    x = reshape(x, size(x, 1), [], numAnchors) ; % -> (H,W,A)
    x = permute(x, [1 3 2]) ; % -> (H, A, W)
    x = reshape(x, [], size(x, 3)) ; % -> (H*A, W)
  end

  fprintf('comparing %s vs %s\n', vName, vName_) ;
  diff = mean(abs(x(:) - x_(:))) ;
  fprintf('diff: %.2f\n', diff) ;
  fprintf('sums: %g vs %g\n', sum(x(:)), sum(x_(:))) ;
end
