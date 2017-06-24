featDir = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/feats') ;
featPath = fullfile(featDir, 'blobs-VGG16.mat') ;
feats = load(featPath) ;

% temp fix
if 1  % chicken dinner
  p = dag.params(dag.getParamIndex('fc6_filter')).value ;
  p = reshape(p, 7,7,[], size(p,4)) ;
  p = permute(p, [2 1 3 4]) ;
  dag.params(dag.getParamIndex('fc6_filter')).value = p ;
end

%dag.layers(dag.getLayerIndex('roi_pool5'))
%dag.layers(dag.getLayerIndex('roi_pool5')).block.flatten = 0 ;

data = permute(feats.data, [3 4 2 1]) ;
data = data(:,:, [3 2 1]) ;
im_info = feats.im_info ;
dag.conserveMemory = 0 ;
dag.eval({'data', data, 'im_info', im_info}) ;

% determine name map
map = containers.Map() ; 
xName = 'data' ;
for ii = 1:numel(dag.layers)
  prev = xName ;
  xName = dag.layers(ii).name ;
  fprintf('%d: processing %s\n', ii, xName) ;
  if findstr('conv', xName), continue; end % only relu outputs are stored
  if findstr('norm', xName), map(xName) = xName; end % norm uses same naming
  if findstr('pool', xName), map(xName) = xName; end % pool uses same naming
  if findstr('relu', xName), map(sprintf('%sx', prev)) = prev ; end
end

keepers = {'rpn_cls_score', 'rpn_bbox_pred', 'rois', 'pool5', ...
           'bbox_pred', 'cls_prob'} ;
for ii = 1:numel(keepers)
  map(keepers{ii}) = keepers{ii} ;
end
%map('rois') = 'rois' ;
  
for ii = 1:numel(dag.vars)
  xName = dag.vars(ii).name ;
  %fprintf('%d: %s\n', ii, xName) ;
  if ~isKey(map, xName), continue ; end 
  xName_ = map(xName) ;

  x = dag.vars(dag.getVarIndex(xName)).value ;
  x_ = feats.(xName_) ;
  x_ = permute(x_, [3 4 2 1]) ;
  diff = x(:) - x_(:) ;
  fprintf('%d: %s vs %s\n', ii, xName, xName_) ;
  fprintf('diff: %g\n', mean(abs(diff))) ;
end
