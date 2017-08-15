function faster_rcnn_train(expDir, opts, varargin) 
%FASTER_RCNN_TRAIN train a Faster R-CNN network end to end

  % load imdb
  if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
  else
    imdb = opts.dataOpts.getImdb(opts) ;
    mkdir(fileparts(opts.dataOpts.imdbPath)) ;
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
  end

  [opts, imdb] = opts.dataOpts.prepareImdb(imdb, opts) ;
  fprintf('finished loading imdb\n') ;

  % train network
  if ~exist(expDir, 'dir'), mkdir(expDir) ; end
  confirmConfig(expDir, opts) ;
  net = opts.modelOpts.net_init(opts) ;

  opts.batchOpts.averageImage = net.meta.normalization.averageImage ;

  [~,~] = cnn_train_autonn(net, imdb, ...
                    @(i,b) opts.modelOpts.get_batch(i, b, opts.batchOpts), ...
                    opts.train, 'expDir', expDir) ;

  % check scores across validation set (handles mAP scoring issue)
  for ii = 20:2:30
    [net, modelName] = deployModel(expDir, opts) ;
    modelName = [modelName sprintf('-%d', ii)] ;
    opts.eval_func('net', net, 'modelName', modelName, 'gpus', opts.train.gpus) ;
  end

% --------------------------------------------------------------
function [net, modelName] = deployModel(expDir, opts)
% --------------------------------------------------------------
  checkpointOpts = {'priorityMetric', 'multitask_loss', 'prune', false} ;
  bestEpoch = findBestEpoch(expDir, checkpointOpts{:}) ;
  bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
  deployPath = sprintf(opts.modelOpts.deployPath, bestEpoch) ;
  opts.modelOpts.deploy_func(bestNet, deployPath) ;
  net = Net(load(deployPath)) ;
  [~,modelName,~] = fileparts(expDir) ; 
