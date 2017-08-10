function net = faster_rcnn_init(opts, varargin)
% FASTER_RCNN_INIT Initialize a Faster R-CNN Detector Network
%   FASTER_RCNN_INIT(OPTS) - constructs a Faster R-CNN detector 
%   according to the options provided using the autonn matconvnet
%   wrapper.

  modelName = opts.modelOpts.architecture ;
  modelDir = fullfile(vl_rootnn, 'data/models-import') ;

  switch modelName
    case 'vgg16'
      trunkPath = fullfile(modelDir, 'imagenet-vgg-verydeep-16.mat') ;
      rootUrl = 'http://www.vlfeat.org/matconvnet/models' ;
      trunkUrl = [rootUrl '/imagenet-vgg-verydeep-16.mat' ] ;
    case 'vgg16-reduced'
      trunkPath = fullfile(modelDir, 'vgg-vd-16-reduced.mat') ;
      rootUrl = 'http://www.robots.ox.ac.uk/~albanie/models' ;
      trunkUrl = [rootUrl '/ssd/vgg-vd-16-reduced.mat'] ;
    case 'resnet50', error('%s not yet supported', modelName) ;
    case 'resnext101_64x4d', error('%s not yet supported', modelName) ;
    otherwise, error('architecture %d is not recognised', modelName) ;
  end

  if ~exist(trunkPath, 'file')
    fprintf('%s not found, downloading... \n', opts.modelOpts.architecture) ;
    mkdir(fileparts(trunkPath)) ; urlwrite(trunkUrl, trunkPath) ;
  end

  net = vl_simplenn_tidy(load(trunkPath)) ; net = dagnn.DagNN.fromSimpleNN(net) ;

  % freeze early layers and modify trunk biases to match caffe
  freeze = {'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'} ;
  for ii = 1:length(freeze)
    pIdx = net.getParamIndex(net.layers(net.getLayerIndex(freeze{ii})).params) ;
    [net.params(pIdx).learningRate] = deal(0) ;
    [net.params(pIdx).weightDecay] = deal(0) ;
  end
  params = {'conv3_1b', 'conv3_2b', 'conv3_3b', 'conv4_1b', 'conv4_2b', ...
          'conv4_3b', 'conv5_1b', 'conv5_2b', 'conv5_3b' 'fc6b', 'fc7b' } ;
  for ii = 1:length(params), net = matchCaffeBiases(net, params{ii}) ; end


  if strcmp(opts.modelOpts.architecture, 'vgg16-reduced') % match caffe
    net.layers(net.getLayerIndex('fc6')).block.dilate = [6 6] ;
    net.layers(net.getLayerIndex('fc6')).block.pad = [6 6] ;
    net.layers(net.getLayerIndex('pool5')).block.stride = [1 1] ;
    net.layers(net.getLayerIndex('pool5')).block.poolSize = [3 3] ;
    net.layers(net.getLayerIndex('pool5')).block.pad = [1 1 1 1] ;
  end
  net.removeLayer('fc8') ; net.removeLayer('prob') ; net.renameVar('x0', 'data') ;


  rng(0) ; % for reproducibility, fix the seed

  % configure autonn inputs
  gtBoxes = Input('gtBoxes') ; 
  gtLabels = Input('gtLabels') ; 
  imInfo = Input('imInfo') ;

  % convert to autonn
  stored = Layer.fromDagNN(net) ; net = stored{1} ;

  % Region proposal network 
  src = net.find('relu5_3', 1) ; 
  largs = {'stride', [1 1], 'pad', [1 1 1 1]} ; sz = [3 3 512 512] ; addRelu = 1 ; 
  rpn_conv = add_block(src, 'rpn_conv_3x3', opts, sz, addRelu, largs{:}) ;
  numAnchors = numel(opts.modelOpts.scales) * numel(opts.modelOpts.ratios) ;

  name = 'rpn_cls_score' ; largs = {'stride', [1 1], 'pad', [0 0 0 0]} ;  
  c = 2 ; sz = [1 1 512 numAnchors*c ] ; addRelu = 0 ;
  rpn_cls = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  name = 'rpn_bbox_pred' ; largs = {'stride', [1 1], 'pad', [0 0 0 0]} ;
  b = 4 ; sz = [1 1 512 numAnchors*b ] ; addRelu = 0 ;
  rpn_bbox_pred = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  largs = {'name', 'rpn_cls_score_reshape'} ;
  args = {rpn_cls, [0 -1 c 0]} ; 
  rpn_cls_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  args = {rpn_cls, gtBoxes, imInfo} ; % note: first input used to determine shape
  largs = {'name', 'anchor_targets', 'numInputDer', 0} ;
  [rpn_labels, rpn_bbox_targets, rpn_iw, rpn_ow, rpn_cw] = ...
                            Layer.create(@vl_nnanchortargets, args, largs{:}) ;

  % rpn losses
  args = {rpn_cls_reshape, rpn_labels, 'instanceWeights', rpn_cw} ;
  largs = {'name', 'rpn_loss_cls', 'numInputDer', 1} ;
  rpn_loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', rpn_iw, 'outsideWeights', rpn_ow} ;
  args = [{rpn_bbox_pred, rpn_bbox_targets, 'sigma', 3}, weighting] ;
  largs = {'name', 'rpn_loss_bbox', 'numInputDer', 1} ;
  rpn_loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {rpn_loss_cls, rpn_loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'rpn_multitask_loss'} ;
  rpn_multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;

  % RoI proposals 
  largs = {'name', 'rpn_cls_prob', 'numInputDer', 0} ;
  rpn_cls_prob = Layer.create(@vl_nnsoftmax, {rpn_cls_reshape}, largs{:}) ;

  args = {rpn_cls_prob, [0 -1 numAnchors*c 0]} ; 
  largs = {'name', 'rpn_cls_prob_reshape', 'numInputDer', 0} ; 
  rpn_cls_prob_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  proposalConf = {'postNMSTopN', 2000, 'preNMSTopN', 12000} ;
  featOpts = [{'featStride', opts.modelOpts.featStride}, proposalConf] ;
  args = {rpn_cls_prob_reshape, rpn_bbox_pred, imInfo, featOpts{:}} ; %#ok
  largs = {'name', 'proposal', 'numInputDer', 0} ; 
  proposals = Layer.create(@vl_nnproposalrpn, args, largs{:}) ;

  args = {proposals, gtBoxes, gtLabels, 'numClasses', opts.modelOpts.numClasses} ;
  largs = {'name', 'roi_data', 'numInputDer', 0} ;
  [rois, labels, bbox_targets, bbox_in_w, bbox_out_w, cw] = ...
                   Layer.create(@vl_nnproposaltargets, args, largs{:}) ;

  % reattach fully connected layers following roipool
  largs = {'name', 'roi_pool5', 'numInputDer', 1} ;
  args = {src, rois, 'method', 'Max', 'Subdivisions', [7,7], 'Transform', 1/16} ;
  roi_pool = Layer.create(@vl_nnroipool, args, largs{:}) ;
  tail = net.find('fc6',1) ; tail.inputs{1} = roi_pool ;

  % insert dropout layers
  relu6 = net.find('relu6', 1) ;
  drop6 = vl_nndropout(relu6, 'rate', 0.5) ; drop6.name = 'drop6' ;
  tail = net.find('fc7',1) ; tail.inputs{1} = drop6 ;

  relu7 = net.find('relu7', 1) ;
  drop7 = vl_nndropout(relu7, 'rate', 0.5) ; drop7.name = 'drop7' ;

  % final predictions
  largs = {'stride', [1 1], 'pad', [0 0 0 0]} ;
  sz = [1 1 4096 opts.modelOpts.numClasses] ; 
  cls_score = add_block(drop7, 'cls_score', opts, sz, 0, largs{:}) ;

  largs = {'stride', [1 1], 'pad', [0 0 0 0]} ;
  sz = [1 1 4096 opts.modelOpts.numClasses*4] ; 
  bbox_pred = add_block(drop7, 'bbox_pred', opts, sz, 0, largs{:}) ;

  % r-cnn losses
  largs = {'name', 'loss_cls', 'numInputDer', 1} ;
  args = {cls_score, labels, 'instanceWeights', cw} ;
  loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', bbox_in_w, 'outsideWeights', bbox_out_w} ;
  args = [{bbox_pred, bbox_targets, 'sigma', 1}, weighting] ;
  largs = {'name', 'loss_bbox', 'numInputDer', 1} ;
  loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {loss_cls, loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'multitask_loss'} ;
  multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;

  checkLearningParams(rpn_multitask_loss, multitask_loss, opts) ;
  net = Net(rpn_multitask_loss, multitask_loss) ;

  % set meta information to match original training code
  rgb = [122.771, 115.9465, 102.9801] ;
  net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;


% ---------------------------------------------------------------------
function net = add_block(net, name, opts, sz, nonLinearity, varargin)
% ---------------------------------------------------------------------

  filters = Param('value', init_weight(sz, 'single', opts), 'learningRate', 1) ;
  biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;
  cudaOpts = {'CudnnWorkspaceLimit', opts.modelOpts.CudnnWorkspaceLimit} ;
  net = vl_nnconv(net, filters, biases, varargin{:}, cudaOpts{:}) ;
  net.name = name ;

  if nonLinearity
    bn = opts.modelOpts.batchNormalization ;
    rn = opts.modelOpts.batchRenormalization ;
    assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
    if bn
      net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
      net.name = sprintf('%s_bn', name) ;
    elseif rn
      net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ; 
      net.name = sprintf('%s_rn', name) ;
    end
    net = vl_nnrelu(net) ;
    net.name = sprintf('%s_relu', name) ;
  end

% ------------------------------------------------
function checkLearningParams(rpn_loss, loss, opts)
% ------------------------------------------------
% compare parameters against caffe.  
% TODO: this check can now be safely disabled

  % create name map
  nameMap = containers.Map ; 
  nameMap('rpn_conv/3x3') = 'rpn_conv_3x3' ;
  proto = fileread(opts.modelOpts.protoPath) ;

  % mini parser
  name = 'blank' ; stack = {} ; tokens = strsplit(proto, '\n') ; prev = '' ;
  assert(contains(tokens{1}, 'VGG_ILSVRC_16'), 'wrong proto') ; tokens(1) = [] ; 
  isBias = 0 ;
  while ~isempty(tokens)
    head = tokens{1} ; tokens(1) = [] ;
    if contains(head, '}') && contains(head, '{') 
      s = strfind(head, '{') ; e = strfind(head, '}') ; head = head(s+1:e-1) ;
    elseif contains(head, '}'), stack(end) = [] ; 
    elseif contains(head, '{'), stack{end+1} = head ; 
    elseif contains(head, 'name') 
      pair = strsplit(head, ':') ; 
      name = strrep(strrep(pair{2}, '"', ''), ' ', '') ;
      if isKey(nameMap, name), name = nameMap(name) ; end
    end

    if ~isempty(stack) && contains(head, {'lr_mult', 'decay'})
      pair = strsplit(head, ':') ; x = str2double(pair{2}) ;
      if contains(head, 'decay') 
        target = 'weightDecay' ; 
      else
        target = 'learningRate' ; 
      end
      if ~strcmp(prev, name)
        fprintf('filter %s (%s): %f \n', target, name, x) ;
      else
        fprintf('bias %s (%s): %f \n', target, name, x) ;
      end
      src = rpn_loss.find(name) ; 
      if isempty(src), src = loss.find(name) ; end
      layer = src{1} ; pos = find(cellfun(@(x) isa(x, 'Param'), layer.inputs)) ;
      param = layer.inputs{pos + isBias} ; val = param.(target) ;
      assert(val == x, sprintf('%s did not match for layer %s', target, name)) ;
      isBias = ~isBias ; prev = name ;
    end
  end

% --------------------------------------------
function weights = init_weight(sz, type, opts)
% --------------------------------------------
% Match caffe fixed scale initialisation, which seems to do 
% better than the Xavier heuristic here

  switch opts.modelOpts.initMethod
    case 'gaussian'
      % in the original code, bounding box regressors are initialised 
      % slightly differently
      numRegressors = opts.modelOpts.numClasses * 4 ;
      if sz(4) ~= numRegressors, sc = 0.01 ; else, sc = 0.001 ; end
    case 'xavier', sc = sqrt(1/(sz(1)*sz(2)*sz(3))) ;
    otherwise, error('%s method not recognised', opts.modelOpts.initMethod) ;
  end
  weights = randn(sz, type)*sc ;

% ----------------------------------------------
function net = matchCaffeBiases(net, param)
% ----------------------------------------------
% set the learning rate and weight decay of the 
% convolution biases to match caffe

  net.params(net.getParamIndex(param)).learningRate = 2 ;
  net.params(net.getParamIndex(param)).weightDecay = 0 ;
