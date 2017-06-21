for ii = 1:numel(dag.layers)
  if isa(dag.layers(ii).block, 'dagnn.Reshape')
    switch dag.layers(ii).name
      case 'rpn_cls_score_reshape'
        dag.layers(ii).block.shape = [-1 0 2] ;
      case 'rpn_cls_prob_reshape'
        dag.layers(ii).block.shape = [-1 0 18] ;
      otherwise
        fprintf('%s\n', dag.layers(ii).name) ;
    end
  end
end

p = dag.params(dag.getParamIndex('fc6_filter')).value ;
p = reshape(p, 7,7,[], size(p,4)) ;
p = permute(p, [2 1 3 4]) ;
dag.params(dag.getParamIndex('fc6_filter')).value = p ;
modPath = '/users/albanie/coding/libs/matconvnets/contrib-matconvnet/data/models-import/faster-rcnn-vggvd-pascal-mods.mat' ;

net = dag.saveobj() ;
save(modPath, '-struct', 'net') ;
