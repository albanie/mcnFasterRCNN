% utility script

featDir = vl_rootnn ;
featPath = fullfile(featDir, 'faster-rcnn-test-blobs.mat') ;
feats = load(featPath) ;

imPath = '~/data/datasets/VOCdevkit2007/VOC2007/JPEGImages/000001.jpg' ;
im = single(imread(imPath)) ;

if ~exist('dag', 'var')
  tmp = load('data/models-import/faster-rcnn-vggvd-pascal-local.mat') ;
  dag = dagnn.DagNN.loadobj(tmp) ;
end

sz = double(size(im)) ; imsz = sz(1:2) ;
sc = 600 ; maxSc = 1000 ; 
factor = max(sc ./ imsz) ; 
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; im_info = [ round(newSz) factor ] ;
imMean = dag.meta.normalization.averageImage ;
im = bsxfun(@minus, im, imMean) ;
data = imresize(im, factor, 'bilinear') ;

dag.conserveMemory = 0 ;
dag.mode = 'test' ;
[dag.params.learningRate] = deal(0) ;
dag.eval({'data', data, 'im_info', im_info}) ;

% determine name map
map = containers.Map() ; 
xName = 'data' ;
for ii = 1:numel(dag.layers)
  prev = xName ;
  xName = dag.layers(ii).name ;
  fprintf('%d: processing %s\n', ii, xName) ;
  if strfind(xName, 'conv'), continue; end % only relu outputs are stored
  if strfind(xName, 'norm'), map(xName) = xName; end % norm uses same naming
  if strfind(xName, 'pool'), map(xName) = xName; end % pool uses same naming
  if strfind(xName, 'relu'), map(sprintf('%sx', prev)) = prev ; end
end

keepers = {'data', 'rpn_cls_score', 'rpn_bbox_pred', 'rois', 'pool5', ...
           'bbox_pred', 'cls_prob'} ;
for ii = 1:numel(keepers)
  map(keepers{ii}) = keepers{ii} ;
end


% trouble
pIdx = dag.getParamIndex(dag.layers(dag.getLayerIndex('conv3_1')).params) ; 
params = {dag.params(pIdx).name} ;

x = dag.vars(dag.getVarIndex('conv3_1')).value ;
xx_ = permute(feats.conv3_1_raw, [3 4 2 1]) ;

% go manual
f = dag.params(pIdx(1)).value ; b = dag.params(pIdx(2)).value ;
  
for ii = 1:numel(dag.vars)
  xName = dag.vars(ii).name ;
  %fprintf('%d: %s\n', ii, xName) ;

  if ~isKey(map, xName), continue ; end 
  xName_ = map(xName) ;

  x = dag.vars(dag.getVarIndex(xName)).value ;
  x_ = feats.(xName_) ;
  x_ = permute(x_, [3 4 2 1]) ;

  switch xName
    case 'data'
      x_ = x_(:,:,[3 2 1]) ;
    case 'rois'
      x_ = squeeze(x_) ; x_ = x_(2:end,:) ; % remove the image index
      x = x(2:end,:) - 1 ; % fix off by one in MATLAB
    %case 'conv3_1x'
      %prev = dag.vars(dag.getVarIndex('pool2')).value ;
      %f_ = permute(f, [2 1 3 4]) ;
      %out_ = vl_nnconv(prev, f, b, 'Pad', 1) ;
      %out = vl_nnrelu(out_) ;
      %diff = sum(abs(out(:)) - abs(x(:))) / sum(abs(out(:))) ;
      %fprintf('diff of out vs x: %g\n', diff) ;
      %diff = sum(abs(out(:)) - abs(x_(:))) / sum(abs(out(:))) ;
      %fprintf('diff of out vs x_: %g\n', diff) ;
      %keyboard
    case 'conv3_1x'
      keyboard
  end


  %diff = x(:) - x_(:) ;
  fprintf('%d: %s vs %s\n', ii, xName, xName_) ;
  diff = sum(abs(x(:)) - abs(x_(:))) / sum(abs(x(:))) ;
  %diff = sum(abs(x(:)) - abs(x_(:))) / sum(abs(x(:))) ;
  fprintf('diff: %g\n', diff) ;
end

