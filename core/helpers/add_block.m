function net = add_block(net, name, opts, sz, nonLinearity, varargin)

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
    in = opts.modelOpts.instanceNormalization ;
    assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
    if bn || in % (instance normalisation will be fixed to train mode later)
      net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
      net.name = sprintf('%s_bn', name) ;
    elseif rn
      net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ; 
      net.name = sprintf('%s_rn', name) ;
    end
    net = vl_nnrelu(net) ;
    net.name = sprintf('%s_relu', name) ;
  end


% --------------------------------------------
function weights = init_weight(sz, type, opts)
% --------------------------------------------
% Match caffe fixed scale initialisation, which seems to do 
% better than the Xavier heuristic here
keyboard % Fix R-FCN vs Faster R-CNN

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

