function net = merge_down_batchnorm(net)
%MERGE_DOWN_BATCHNORM - merge batch norm layers in network
%   NET = MERGE_DOWN_BATCHNORM(NET) iterates through the layers of a DagNN
%   network object and "merges" each batch normalization layer into the
%   previous convolutional layer (using a simple arithmetic transformation).
%   This can be useful, for instances, in cases in which a network must be 
%   used with small batch sizes (making batch normalization inappropriate).
%   (based on cnn_imagenet_deploy.m)
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  bnLayers = arrayfun(@(x) isa(x.block, 'dagnn.BatchNorm'), net.layers) ;
  names = arrayfun(@(x) {x.name}, net.layers(bnLayers)) ;
  for ii = 1:numel(names)
    layerName = names{ii} ;
    layer = net.layers(net.getLayerIndex(layerName)) ;

    % merge into previous conv layer
    playerName = dagFindLayersWithOutput(net, layer.inputs{1}) ;
    playerName = playerName{1} ;
    playerIndex = net.getLayerIndex(playerName) ;
    player = net.layers(playerIndex) ;
    msg = 'bnorm cannot be merged as it is not preceded by a conv layer.' ;
    if ~isa(player.block, 'dagnn.Conv'), error(msg) ; end

    % if the convolution layer does not have a bias recreate it to have one
    if ~player.block.hasBias
      block = player.block ; block.hasBias = true ;
      net.renameLayer(playerName, 'tmp') ;
      pNames = {player.params{1}, sprintf('%s_b',playerName)} ;
      largs = {playerName, block, player.inputs, player.outputs, pNames} ;
      net.addLayer(largs{:}) ; net.removeLayer('tmp') ;
      playerIndex = net.getLayerIndex(playerName) ;
      player = net.layers(playerIndex) ;
      biases = net.getParamIndex(player.params{2}) ;
      net.params(biases).value = zeros(block.size(4), 1, 'single') ;
    end

    filters = net.getParamIndex(player.params{1}) ;
    biases = net.getParamIndex(player.params{2}) ;
    gains = net.getParamIndex(layer.params{1}) ;
    offsets = net.getParamIndex(layer.params{2}) ;
    moments = net.getParamIndex(layer.params{3}) ;
    pIdx = [filters, biases, gains, offsets, moments] ;
    pVals = arrayfun(@(x) {net.params(x).value}, pIdx) ;
    [filtersValue, biasesValue] = mergeParams(pVals{:}) ;
    net.params(filters).value = filtersValue ;
    net.params(biases).value = biasesValue ;
    fprintf('merging %s into previous conv \n', layer.name) ;
  end
  dagRemoveLayersOfType(net, 'dagnn.BatchNorm') ;

% -------------------------------------------------------------------------
function [filters, biases] = mergeParams(filters, biases, gains, offsets, moments)
% -------------------------------------------------------------------------
% (copied from cnn_imagenet_deploy.m)
  % wk / sqrt(sigmak^2 + eps)
  % bk - wk muk / sqrt(sigmak^2 + eps)
  a = gains(:) ./ moments(:,2) ;
  b = offsets(:) - moments(:,1) .* a ;
  biases(:) = biases(:) + b(:) ;
  sz = size(filters) ;
  numFilters = sz(4) ;
  filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;

% -------------------------------------------------------------------------
function layers = dagFindLayersWithOutput(net, outVarName)
% -------------------------------------------------------------------------
  layers = {} ;
  for l = 1:numel(net.layers)
    if any(strcmp(net.layers(l).outputs, outVarName))
      layers{1,end+1} = net.layers(l).name ; %#ok
    end
  end

% -------------------------------------------------------------------------
function dagRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
  names = dagFindLayersOfType(net, type) ;
  for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end

% -------------------------------------------------------------------------
function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
  layers = [] ;
  for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
      layers{1,end+1} = net.layers(l).name ; %#ok
    end
  end
