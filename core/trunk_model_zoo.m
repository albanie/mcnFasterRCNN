function dag = trunk_model_zoo(modelName, opts)

  modelDir = fullfile(vl_rootnn, 'data/models-import') ;
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
    fprintf('%s not found, downloading... \n', modelName) ;
    mkdir(fileparts(trunkPath)) ; urlwrite(trunkUrl, trunkPath) ;
  end

  storedNet = load(trunkPath) ;
  if ~isfield(storedNet, 'vars') % check for dagnn
    net = vl_simplenn_tidy(storedNet) ; dag = dagnn.DagNN.fromSimpleNN(net) ;
  else
    dag = dagnn.DagNN.loadobj(storedNet) ;
  end

  % prune unused layers
  switch modelName
    case 'vgg16'
      dag.removeLayer('fc8') ; 
      dag.renameVar('x0', 'data') ; % fix old naming scheme
    case 'resnet50'
      dag.removeLayer('fc1000') ; 
  end
  dag.removeLayer('prob') ;  

  % flatten bnorm if required
  if opts.modelOpts.mergeBnorm, dag = merge_down_batchnorm(dag) ; end
