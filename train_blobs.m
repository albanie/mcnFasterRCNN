full = Layer.fromCompiledNet(net) ;
%tail_cls = full{1}.find('rpn_loss_cls',1) ;
%tail_bbox = full{1}.find('rpn_loss_bbox',1) ;
tail_rpn_loss = full{1}.find('rpn_multitask_loss', 1) ;
tail_frcnn_loss = full{2}.find('multitask_loss', 1) ;


tmp = tail_frcnn_loss.find('fc6', 1).inputs{2} ; 
tmp = reshape(tmp, 7, 7, [], size(tmp, 4)) ; tmp = permute(tmp, [2 1 3 4]) ; 
tail_frcnn_loss.find('fc6',1).inputs{2} = tmp ;
% squeeze bias
%tail.find('conv1_1',1).inputs{3}.value = squeeze(tail.find('conv1_1',1).inputs{3}.value) ;

%tail.find('conv1_1',1).inputs{2}.value = permute(tail.find('conv1_1',1).inputs{2}.value, [ 1 2 3 4]) ;

%f = tail.find('conv1_1',1).inputs{2}.value ;
%b = tail.find('conv1_1',1).inputs{3}.value ;

% set bias to zero
%tail.find('conv1_1',1).inputs{3}.value = zeros(size(tail.find('conv1_1',1).inputs{3}.value), 'single') ;

tailNet = Net(tail_rpn_loss, tail_frcnn_loss) ;

im = permute(store.data, [3 4 2 1]) ; im = im(:,:, [3 2 1]) ;
l = {store.gt_boxes(5) + 1} ; 
b = {store.gt_boxes(1:4) + 1} ; 
imInfo = store.im_info ;
batchData = {'data', im, 'gtBoxes', b, 'gtLabels', l, 'imInfo', imInfo} ;
%batchData = {'data', im} ;
tailNet.eval(batchData, 'forward') ;

% check feats
%info = tailNet.getVarInfo() ;
%keep = find(cellfun(@(x) strcmp(x, 'layer'), {info.type})) ;
sample = {'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'conv5_3', ...
          'rpn_cls_score', 'rpn_bbox_pred', ...
          'rpn_cls_score_reshape', 'rpn_labels', 'rpn_bbox_targets', ...
          'rpn_bbox_inside_weights', 'rpn_bbox_outside_weights', ...
          'rpn_loss_bbox', 'rpn_loss_cls', 'proposals', 'rois', ...
          'roi_pool5', 'fc6', 'fc7', 'cls_score', 'bbox_pred', ...
          'bbox_inside_weights', 'bbox_outside_weights', 'bbox_targets',...
          'loss_cls', 'loss_bbox'} ;

offset = 0 ;
for ii = 1:numel(sample)
  vName = sample{ii} ; vName_ = vName ;
  tokens = regexp(vName, 'relu(\d[_]\d)', 'tokens') ;
  if ~isempty(tokens), vName_ = sprintf('conv%s', tokens{1}{1}) ; end
  if strcmp(vName, 'rpn_bbox_loss'), vName_ = 'rpn_loss_bbox' ; end
  if strcmp(vName, 'rpn_loss_cls'), vName_ = 'rpn_cls_loss' ; end
  if strcmp(vName, 'proposals'), vName_ = 'rpn_rois' ; end
  if strcmp(vName, 'roi_pool5'), vName_ = 'pool5' ; end
  %if strcmp(vName, 'rois'), vName_ = 'rois' ; end

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

  switch vName
    case 'bbox_targets' 
      offset = 3 ; vName = 'roi_data' ;
    case 'bbox_inside_weights' 
      offset = 4 ; vName = 'roi_data' ;
    case 'bbox_outside_weights' 
      offset = 5 ; vName = 'roi_data' ;
  end

  switch vName
    case 'rois'
      offset = 1 ; vName = 'roi_data' ;
  end

  x = tailNet.getValue(vName) ; 
  x_ = permute(store.(vName_), [3 4 2 1]) ;
  if ismember(vName, {'rpn_labels', 'roi_data'}), x = x{offset} ; end

  %if strcmp(vName_, 'bbox_inside_weights'), keyboard ; end
  %if strcmp(vName, 'rpn_cls_score_reshape'), keyboard ; end
  %if strcmp(vName, 'rpn_loss_cls'), keyboard ; end


  if offset == 1 && strcmp(vName, 'rpn_labels')
    % to account for column major ordering, we are using a different 
    % label shape i.e. Hx(WxA)x1x1 vs 1x1x(AxH)xW where H is faster than A in caffe
    numAnchors = 9 ;

    % match caffe convention of -1 ignore, 0 negative
    x(x == -1) = -2 ; x(x == 0) = -1 ; x(x == -2) = 0 ;

    x = reshape(x, size(x, 1), [], numAnchors) ; % -> (H,W,A)
    x = permute(x, [1 3 2]) ; % -> (H, A, W)
    x = reshape(x, [], size(x, 3)) ; % -> (H*A, W)
  end
  if strcmp(vName, 'proposals'), x = x - 1 ; 
    x_ = squeeze(x_) ; 
  end
  if strcmp(vName, 'roi_data') && offset == 1 
    x_ = squeeze(x_)' ; x = x - 1 ;
  end
  if strcmp(vName, 'roi_pool5') 
    z1 = tailNet.getValue('roi_data') ; z1 = z1{1}' ;
    z2 = tailNet.getValue('relu5_3') ;
    z3 = tailNet.getValue('roi_pool5') ;

    y1 = (store.rois + 1)' ;
    y2 = permute(store.conv5_3, [3 4 2 1]) ;
    y3 = permute(store.pool5, [3 4 2 1]) ;

    res1 = vl_nnroipool(z2, z1, 'subdivisions', [7,7], 'transform', 1/16) ;
    res2 = vl_nnroipool(z2, z1, 'Subdivisions', [7,7], 'Transform', 1/16) ;

    %cRes = permute(store.pool5, [3 4 2 1]) ;
  end

  if strcmp(vName, 'loss_bbox')
    tmp = tailNet.getValue('roi_data') ; 
    iw1 = tmp{4} ; ow1 = tmp{5} ;
    t1 = tmp{3} ; 
    b1 = tailNet.getValue('bbox_pred') ;
    z1 = vl_nnsmoothL1loss(b1, t1, 'sigma', 1, ...
                 'insideWeights', iw1, 'outsideWeights', ow1) ;

    iw2 = store.bbox_inside_weights ; ow2 = store.bbox_outside_weights ;
    b2 = store.bbox_pred ;
    y1 = store.loss_bbox ;
    %keyboard
  end
  fprintf('comparing %s vs %s\n', vName, vName_) ;
  %diff = mean(abs(x(:) - x_(:))) ;
  diff = norm(x(:)-x_(:)) / norm(x(:)) ;
  fprintf('diff: %.2f\n', diff) ;
  fprintf('sums: %g vs %g\n', sum(x(:)), sum(x_(:))) ;
end
