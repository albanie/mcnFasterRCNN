function net = freeze_and_match_layers(net, freezeDecay, opts)

  modelName = opts.modelOpts.architecture ;
  switch modelName
    case 'vgg16'
      % freeze early layers and modify trunk biases to match caffe
      biases = {'conv3_1b', 'conv3_2b', 'conv3_3b', 'conv4_1b', 'conv4_2b', ...
              'conv4_3b', 'conv5_1b', 'conv5_2b', 'conv5_3b' 'fc6b', 'fc7b' } ;
      freeze = {'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'} ;
    case 'resnet50'
      if opts.modelOpts.atrous
        % Unit 5 of the resnet is modified slightly for detection in the r-fcn
        net.layers(net.getLayerIndex('res5a_branch1')).block.stride = [1 1] ;
        net.layers(net.getLayerIndex('res5a_branch2a')).block.stride = [1 1] ;
        dilateLayers = {'res5a_branch2b', 'res5b_branch2b', 'res5c_branch2b' } ;
        for ii = 1:numel(dilateLayers)
          lIdx = net.getLayerIndex(dilateLayers{ii}) ;
          net.layers(lIdx).block.dilate = [2 2] ;
          net.layers(lIdx).block.pad = [2 2 2 2] ;
        end
      end
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


% -----------------------------------------
function net = matchCaffeBiases(net, param)
% -----------------------------------------
% set the learning rate and weight decay of the 
% convolution biases to match caffe

  net.params(net.getParamIndex(param)).learningRate = 2 ;
  net.params(net.getParamIndex(param)).weightDecay = 0 ;
