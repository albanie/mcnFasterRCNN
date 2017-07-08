full = Layer.fromCompiledNet(net) ;
tail_cls = full{1}.find('rpn_loss_cls',1) ;
tail_bbox = full{1}.find('rpn_loss_bbox',1) ;

% squeeze bias
%tail.find('conv1_1',1).inputs{3}.value = squeeze(tail.find('conv1_1',1).inputs{3}.value) ;

%tail.find('conv1_1',1).inputs{2}.value = permute(tail.find('conv1_1',1).inputs{2}.value, [ 1 2 3 4]) ;

%f = tail.find('conv1_1',1).inputs{2}.value ;
%b = tail.find('conv1_1',1).inputs{3}.value ;

% set bias to zero
%tail.find('conv1_1',1).inputs{3}.value = zeros(size(tail.find('conv1_1',1).inputs{3}.value), 'single') ;

tailNet = Net(tail_cls, tail_bbox) ;

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
      offset = 4 ; vName = 'rpn_labels' ;
  end

  x = tailNet.getValue(vName) ; 
  x_ = permute(store.(vName_), [3 4 2 1]) ;

  if strcmp(vName, 'rpn_labels'), x = x{offset} ; end
  fprintf('comparing %s vs %s\n', vName, vName_) ;
  diff = mean(abs(x(:) - x_(:))) ;
  fprintf('diff: %.2f\n', diff) ;
  fprintf('sums: %g vs %g\n', sum(x(:)), sum(x_(:))) ;
end
