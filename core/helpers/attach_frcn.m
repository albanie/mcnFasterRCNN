function frcn_out = attach_frcn(net, frcn, rpn_out, opts)

  % reattach fully connected layers following roipool
  src = net.find(frcn.base, 1) ; 
  largs = {'name', 'roi_pool5', 'numInputDer', 1} ;
  args = {src, rpn_out.rois, 'method', 'max', ...
          'subdivisions', frcn.divisions, 'transform', 1/16} ;
  roi_pool = Layer.create(@vl_nnroipool, args, largs{:}) ;
  for ii = 1:numel(frcn.heads)
    body = net.find(frcn.heads{ii}, 1) ; body.inputs{1} = roi_pool ;
  end

  if frcn.insert_dropout % only used by VGG 16
    msg = 'only vgg16-based architecture uses dropout' ;
    assert(strcmp(opts.modelOpts.architecture, 'vgg16'), msg) ;
    relu6 = net.find('relu6', 1) ;
    drop6 = vl_nndropout(relu6, 'rate', 0.5) ; drop6.name = 'drop6' ;
    tail = net.find('fc7',1) ; tail.inputs{1} = drop6 ;
    relu7 = net.find('relu7', 1) ;
    tail = vl_nndropout(relu7, 'rate', 0.5) ; tail.name = 'drop7' ;
  else
    tail = net.find(frcn.tail, 1) ;
  end

  % final predictions
  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  sz = [1 1 frcn.channels_in opts.modelOpts.numClasses] ; 
  cls_score = add_conv_block(tail, 'cls_score', opts, sz, 0, largs{:}) ;

  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  sz = [1 1 frcn.channels_in opts.modelOpts.numClasses*4] ; 
  bbox_pred = add_conv_block(tail, 'bbox_pred', opts, sz, 0, largs{:}) ;

  % r-cnn losses
  largs = {'name', 'loss_cls', 'numInputDer', 1} ;
  args = {cls_score, rpn_out.labels, 'instanceWeights', rpn_out.cw} ;
  loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', rpn_out.bbox_in_w, ...
               'outsideWeights', rpn_out.bbox_out_w} ;
  args = [{bbox_pred, rpn_out.bbox_targets, 'sigma', 1}, weighting] ;
  largs = {'name', 'loss_bbox', 'numInputDer', 1} ;
  loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {loss_cls, loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'multitask_loss'} ;
  multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;

  % add a generic accuracy tracker to r-cnn for extra clarity
  largs = {'name', 'cls_error', 'numInputDer', 1} ;
  clsOpts = {'loss', 'classerror', 'instanceWeights', rpn_out.cw} ;
  args = [{cls_score, rpn_out.labels} clsOpts] ;
  cls_error = Layer.create(@vl_nnloss, args, largs{:}) ;

  frcn_out.cls_error = cls_error ;
  frcn_out.multitask_loss = multitask_loss ;
