function net = faster_rcnn_init(opts, varargin)
% FASTER_RCNN_INIT Initialize a Faster R-CNN Detector Network
%   FASTER_RCNN_INIT(OPTS) - constructs a Faster R-CNN detector 
%   according to the options provided using the autonn matconvnet
%   wrapper.

  rng(0) ; % for reproducibility, fix the seed

  % configure autonn inputs
  gtBoxes = Input('gtBoxes') ; 
  gtLabels = Input('gtLabels') ; 
  imInfo = Input('imInfo') ;

  % select RPN/fast-rcnn locations
  switch opts.modelOpts.architecture
    case 'vgg16'
      rpn_base = 'relu5_3' ;
      rpn_channels_in = 512 ;
      rpn_channels_out = 512 ;
      fast_rcnn_channels_in = 4096 ;
      fast_rcnn_heads = {'fc6'} ;
      insert_dropout = true ;
      divisions = [7, 7] ;
      freeze_decay = 1 ;
      fast_rcnn_tail = 'drop7' ;
    case 'resnet50'
      rpn_base = 'res4f_relu' ;
      fast_rcnn_heads = {'res5a_branch1', 'res5a_branch2a'} ;
      rpn_channels_in = 1024 ;
      rpn_channels_out = 256 ;
      fast_rcnn_channels_in = 2048 ;
      divisions = [14, 14] ;
      insert_dropout = false ;
      freeze_decay = 0 ;
      fast_rcnn_tail = 'pool5' ;
  end

  % freeze early layers and modify trunk biases to match caffe
  dag = loadTrunkModel(opts) ;
  dag = freezeAndMatchLayers(dag, freeze_decay, opts) ;
  dag = pruneUnusedLayers(dag, opts) ;

  % flatten bnorm if required
  if opts.modelOpts.mergeBnorm, dag = merge_down_batchnorm(dag) ; end

  % convert to autonn
  stored = Layer.fromDagNN(dag) ; net = stored{1} ;

  % Region proposal network 
  src = net.find(rpn_base, 1) ; 
  largs = {'stride', [1 1], 'pad', [1 1 1 1], 'dilate', [1 1]} ; 
  sz = [3 3 rpn_channels_in rpn_channels_out] ; addRelu = 1 ; 
  rpn_conv = add_block(src, 'rpn_conv_3x3', opts, sz, addRelu, largs{:}) ;
  numAnchors = numel(opts.modelOpts.scales) * numel(opts.modelOpts.ratios) ;

  name = 'rpn_cls_score' ; c = 2 ; sz = [1 1 rpn_channels_out numAnchors*c ] ; 
  addRelu = 0 ; largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;  
  rpn_cls = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  name = 'rpn_bbox_pred' ; b = 4 ; sz = [1 1 rpn_channels_out numAnchors*b ] ;
  addRelu = 0 ;
  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
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
  args = {src, rois, 'method', 'Max', 'Subdivisions', divisions, 'Transform', 1/16} ;
  roi_pool = Layer.create(@vl_nnroipool2, args, largs{:}) ;
  for ii = 1:numel(fast_rcnn_heads)
    body = net.find(fast_rcnn_heads{ii}, 1) ; body.inputs{1} = roi_pool ;
  end

  if insert_dropout % only used by VGG 16
    msg = 'only vgg16-based architecture uses dropout' ;
    assert(strcmp(opts.modelOpts.architecture, 'vgg16'), msg) ;
    relu6 = net.find('relu6', 1) ;
    drop6 = vl_nndropout(relu6, 'rate', 0.5) ; drop6.name = 'drop6' ;
    tail = net.find('fc7',1) ; tail.inputs{1} = drop6 ;
    relu7 = net.find('relu7', 1) ;
    tail = vl_nndropout(relu7, 'rate', 0.5) ; tail.name = 'drop7' ;
  else
    tail = net.find(fast_rcnn_tail, 1) ;
  end

  % final predictions
  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  sz = [1 1 fast_rcnn_channels_in opts.modelOpts.numClasses] ; 
  cls_score = add_block(tail, 'cls_score', opts, sz, 0, largs{:}) ;

  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  sz = [1 1 fast_rcnn_channels_in opts.modelOpts.numClasses*4] ; 
  bbox_pred = add_block(tail, 'bbox_pred', opts, sz, 0, largs{:}) ;

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

  fOpts = {'learningRate', 1, 'weightDecay', 1} ;
  filters = Param('value', init_weight(sz, 'single', opts), fOpts{:}) ;
  bOpts = {'learningRate', 2, 'weightDecay', 0} ;
  biases = Param('value', zeros(sz(4), 1, 'single'), bOpts{:}) ;
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

% ---------------------------------------------------------
function net = freezeAndMatchLayers(net, freezeDecay, opts)
% ---------------------------------------------------------

  modelName = opts.modelOpts.architecture ;
  switch modelName
    case 'vgg16'
      % freeze early layers and modify trunk biases to match caffe
      biases = {'conv3_1b', 'conv3_2b', 'conv3_3b', 'conv4_1b', 'conv4_2b', ...
              'conv4_3b', 'conv5_1b', 'conv5_2b', 'conv5_3b' 'fc6b', 'fc7b' } ;
      freeze = {'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'} ;
    case 'resnet50'
      % Unit 5 of the resnet is modified slightly for detection
      %net.layers(net.getLayerIndex('res5a_branch1')).block.stride = [1 1] ;
      %net.layers(net.getLayerIndex('res5a_branch2a')).block.stride = [1 1] ;
      %dilateLayers = {'res5a_branch2b', 'res5b_branch2b', 'res5c_branch2b' } ;
      %for ii = 1:numel(dilateLayers)
        %lIdx = net.getLayerIndex(dilateLayers{ii}) ;
        %net.layers(lIdx).block.dilate = [2 2] ;
        %net.layers(lIdx).block.pad = [2 2 2 2] ;
      %end
      biases = {'conv1_bias'} ;

      % modify padding on pooling layers
      net.layers(net.getLayerIndex('pool1')).block.pad = [0 0 0 0] ;

      base = {'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'} ;
      leaves = {'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'} ;
      template = 'res2%s_branch2%s' ;
      resUnits = cellfun(@(x,y) {sprintf(template, x,y)}, base,leaves) ;
      freeze = [{'conv1', 'res2a_branch1'}, resUnits] ;
    case 'resnext101_64x4d', error('%s not yet supported', modelName) ;
    otherwise, error('architecture %d is not recognised', modelName) ;
  end
  for ii = 1:length(biases), net = matchCaffeBiases(net, biases{ii}) ; end

  for ii = 1:length(freeze)
    pIdx = net.getParamIndex(net.layers(net.getLayerIndex(freeze{ii})).params) ;
    [net.params(pIdx).learningRate] = deal(0) ;
    % In the original code weight decay is kept on in the conv layers
    if freezeDecay
      [net.params(pIdx).weightDecay] = deal(0) ;
    end
  end

  % regardless of model, freeze all batch norms during learning
  bnLayerIdx = find(arrayfun(@(x) isa(x.block, 'dagnn.BatchNorm'), net.layers)) ;
  for ii = 1:length(bnLayerIdx)
    lIdx = bnLayerIdx(ii) ;
    pIdx = net.getParamIndex(net.layers(lIdx).params) ;
    [net.params(pIdx).learningRate] = deal(0) ;
    [net.params(pIdx).weightDecay] = deal(0) ;
  end

% ------------------------------------------
function net = pruneUnusedLayers(net, opts)
% -----------------------------------------
  switch opts.modelOpts.architecture
    case 'vgg16'
      net.removeLayer('fc8') ; 
      net.renameVar('x0', 'data') ; % fix old naming scheme
    case 'resnet50'
      net.removeLayer('fc1000') ; 
    otherwise, error('%s method not recognised', opts.modelOpts.architecture) ;
  end
  net.removeLayer('prob') ;  

% ---------------------------------
function dag = loadTrunkModel(opts)
% ---------------------------------
  modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  modelName = opts.modelOpts.architecture ;
  switch modelName
    case 'vgg16'
      trunkPath = fullfile(modelDir, 'imagenet-vgg-verydeep-16.mat') ;
      rootUrl = 'http://www.vlfeat.org/matconvnet/models' ;
      trunkUrl = [rootUrl '/imagenet-vgg-verydeep-16.mat'] ;
    case 'resnet50'
      trunkPath = fullfile(modelDir, 'imagenet-resnet-50-dag.mat') ;
      rootUrl = 'http://www.vlfeat.org/matconvnet/models' ;
      trunkUrl = [rootUrl 'imagenet-resnet-50-dag.mat'] ;
    case 'resnext101_64x4d', error('%s not yet supported', modelName) ;
    otherwise, error('architecture %d is not recognised', modelName) ;
  end

  if ~exist(trunkPath, 'file')
    fprintf('%s not found, downloading... \n', opts.modelOpts.architecture) ;
    mkdir(fileparts(trunkPath)) ; urlwrite(trunkUrl, trunkPath) ;
  end

  storedNet = load(trunkPath) ;
  if ~isfield(storedNet, 'vars') % check for dagnn
    net = vl_simplenn_tidy(storedNet) ; dag = dagnn.DagNN.fromSimpleNN(net) ;
  else
    dag = dagnn.DagNN.loadobj(storedNet) ;
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
