function fixer(varargin)
%FIXER - address issues with imported caffe models
%  FIXER is desgined to resolve any remaining incompatibilities
%  with layer names and filter layouts that have not been addressed
%  by the model importer
%
%  NOTE: Due to the filter layout re-ordering, this function 
%  is not idempotent
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.modelName = 'faster-rcnn-vggvd-pascal-local.mat' ;
  opts.numAnchors = 9 ; % 12 for coco
  opts = vl_argparse(opts, varargin) ;

  srcPath = fullfile(vl_rootnn, 'data/models-import', opts.modelName) ;
  dag = dagnn.DagNN.loadobj(load(srcPath)) ;

  pIdx = dag.getLayerIndex('proposal') ; % fix proposal layer opt types
  scales = double(dag.layers(pIdx).block.scales) ;
  dag.layers(pIdx).block.scales = reshape(scales, 1, [])  ;
  dag.layers(pIdx).block.featStride = double(dag.layers(pIdx).block.featStride) ;

  for ii = 1:numel(dag.layers)
    if isa(dag.layers(ii).block, 'dagnn.Reshape')
      switch dag.layers(ii).name
        case 'rpn_cls_score_reshape'
          dag.layers(ii).block.shape = [-1 0 2] ;
        case 'rpn_cls_prob_reshape'
          dag.layers(ii).block.shape = [-1 0 2 * opts.numAnchors] ;
        otherwise
          fprintf('%s\n', dag.layers(ii).name) ;
      end
    end
  end

  % fix filter layout
  p = dag.params(dag.getParamIndex('fc6_filter')).value ;
  p = reshape(p, 7,7,[], size(p,4)) ;
  p = permute(p, [2 1 3 4]) ;
  dag.params(dag.getParamIndex('fc6_filter')).value = p ;
  net = dag.saveobj() ; save(srcPath, '-struct', 'net') ; %#ok
