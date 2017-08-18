function layers = apply_instance_normalization(layers)

  % fix to training mode for IN
  for ii = 1:numel(layers)
    layer = layers{ii} ;
    bn_layers = layer.find(@vl_nnbnorm_wrapper) ;
    for jj = 1:numel(bn_layers)
      ins = bn_layers{jj}.inputs ;
      pos = cellfun(@(x) isa(x, 'Input') && strcmp(x.name, 'testMode'), ins) ;
      if any(pos), bn_layers{jj}.inputs{pos} = 0 ; end
    end
  end
