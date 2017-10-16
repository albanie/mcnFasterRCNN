function net = compile_detector(rpn_out, det_out, meta, opts)

  if opts.checkAgainstProto
    checkLearningParams({rpn_out.multitask_loss, det_out.multitask_loss}, opts) ;
  end

  if opts.modelOpts.mergeBnorm
    bn_layers = rpn_out.multitask_loss.find(@vl_nnbnorm_wrapper) ;
    assert(numel(bn_layers) == 0, 'batch norm should be removed') ;
    bn_layers = det_out.multitask_loss.find(@vl_nnbnorm_wrapper) ;
    assert(numel(bn_layers) == 0, 'batch norm should be removed') ;
  end

  % turn on instance normalisation if requested. This is done by ensuring 
  % a batch size of one, and fixing bn to train mode
  loss_layers = {rpn_out.multitask_loss, det_out.multitask_loss} ;
  if opts.modelOpts.instanceNormalization
    loss_layers = apply_instance_normalization(loss_layers) ;
    assert(opts.modelOpts.batchSize == 1, 'single batch items for IN') ;
  end

  net = Net(loss_layers{:}, det_out.cls_error) ;

  % set meta information to match original training code
  rgb = mean(mean(meta.normalization.averageImage,1),2) ;
  %rgb = [122.771, 115.9465, 102.9801] ;
  net.meta.normalization.averageImage = permute(rgb, [2 3 1]) ;
