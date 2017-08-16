function outputs = attach_rpn(net, rpn_base, opts) 

  % Region proposal network 
  src = net.find(rpn_base, 1) ; 
  largs = {'stride', [1 1], 'pad', [1 1 1 1], 'dilate', [1 1]} ; 
  sz = [3 3 rpn_channels_in rpn_channels_out] ; addRelu = 1 ; 
  rpn_conv = add_block(src, 'rpn_conv_3x3', opts, sz, addRelu, largs{:}) ;
  numAnchors = numel(opts.modelOpts.scales) * numel(opts.modelOpts.ratios) ;

  name = 'rpn_cls_score' ; c = 2 ; sz = [1 1 rpn_channels_out numAnchors*c ] ; 
  addRelu = 0 ; largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;  
  rpn_cls = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  name = 'rpn_bbox_pred' ; b = 4 ; sz = [1 1 rpn_channels_out numAnchors*b ] ;
  addRelu = 0 ;
  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  rpn_bbox_pred = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  largs = {'name', 'rpn_cls_score_reshape'} ;
  args = {rpn_cls, [0 -1 c 0]} ; 
  rpn_cls_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  args = {rpn_cls, gtBoxes, imInfo} ; % note: first input used to determine shape
  largs = {'name', 'anchor_targets', 'numInputDer', 0} ;
  [rpn_labels, rpn_bbox_targets, rpn_iw, rpn_ow, rpn_cw] = ...
                          Layer.create(@vl_nnanchortargets, args, largs{:}) ;

  % rpn losses
  args = {rpn_cls_reshape, rpn_labels, 'instanceWeights', rpn_cw} ;
  largs = {'name', 'rpn_loss_cls', 'numInputDer', 1} ;
  rpn_loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', rpn_iw, 'outsideWeights', rpn_ow} ;
  args = [{rpn_bbox_pred, rpn_bbox_targets, 'sigma', 3}, weighting] ;
  largs = {'name', 'rpn_loss_bbox', 'numInputDer', 1} ;
  rpn_loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {rpn_loss_cls, rpn_loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'rpn_multitask_loss'} ;
  rpn_multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;

  % RoI proposals 
  largs = {'name', 'rpn_cls_prob', 'numInputDer', 0} ;
  rpn_cls_prob = Layer.create(@vl_nnsoftmax, {rpn_cls_reshape}, largs{:}) ;

  args = {rpn_cls_prob, [0 -1 numAnchors*c 0]} ; 
  largs = {'name', 'rpn_cls_prob_reshape', 'numInputDer', 0} ; 
  rpn_cls_prob_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  proposalConf = {'postNMSTopN', 2000, 'preNMSTopN', 12000} ;
  featOpts = [{'featStride', opts.modelOpts.featStride}, proposalConf] ;
  args = {rpn_cls_prob_reshape, rpn_bbox_pred, imInfo, featOpts{:}} ; %#ok
  largs = {'name', 'proposal', 'numInputDer', 0} ; 
  proposals = Layer.create(@vl_nnproposalrpn, args, largs{:}) ;

  args = {proposals, gtBoxes, gtLabels, 'numClasses', opts.modelOpts.numClasses} ;
  largs = {'name', 'roi_data', 'numInputDer', 0} ;
  [rois, labels, bbox_targets, bbox_in_w, bbox_out_w, cw] = ...
                   Layer.create(@vl_nnproposaltargets, args, largs{:}) ;
  
   % store outputs
   outputs.rois = rois ;
   outputs.labels = labels ;
   outputs.bbox_targets = bbox_targets ;
   outputs.bbox_in_w = bbox_in_w ;
   outputs.bbox_out_w = bbox_out_w ;
   outputs.cw = cw ;
   outputs.rpn_multitask_loss = rpn_multitask_loss ;
