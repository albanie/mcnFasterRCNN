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
[~,~] = cnn_train_autonn(net, imdb, ...
                    @(i,b) opts.modelOpts.get_batch(i, b, opts.batchOpts), ...
                    opts.train, 'expDir', expDir) ;


% evaluate network
[net, modelName] = deployModel(expDir, opts) ;
opts.eval_func('net', net, 'modelName', modelName, 'gpus', opts.train.gpus) ;

% ---------------------------------------------------
function [net, modelName] = deployModel(expDir, opts)
% ---------------------------------------------------
checkpointOpts = {'priorityMetric', 'mbox_loss', 'prune', true} ;
bestEpoch = findBestEpoch(expDir, checkpointOpts{:}) ;
bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
deployPath = sprintf(opts.modelOpts.deployPath, bestEpoch) ;
opts.modelOpts.deploy_func(bestNet, deployPath, opts.modelOpts.numClasses) ;
net = dagnn.DagNN.loadobj(load(deployPath)) ;
[~,modelName,~] = fileparts(expDir) ; 
