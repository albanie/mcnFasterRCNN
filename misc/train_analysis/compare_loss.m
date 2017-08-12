function compare_loss(varargin)

  opts.disp = 'combined' ;
  opts.numEpochs = 14 ; % used during mcn training
  %opts.expDir = 'faster-rcnn-pascal-vt-1-vgg16-flip-lr-0.0001-0.001' ;
  %opts.expDir = 'faster-rcnn-pascal-vt-1-vgg16-flip-lr-0.0002-0.002' ;
  opts.expDir = 'faster-rcnn-pascal-vt-1-vgg16-flip-lr-0.0001-0.001-gaussian' ;
  %opts.expDir = 'faster-rcnn-pascal-vt-1-vgg16-flip' ;
  opts.logDir = '~/coding/libs/py-faster-rcnn2/experiments/logs' ;
  opts.figDir = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/train_analysis/figs') ;
  opts = vl_argparse(opts, varargin) ;

  % load caffe training artefacts
  loss_cls = load(fullfile(opts.logDir, 'loss_cls.txt')) ; 
  loss_bbox = load(fullfile(opts.logDir, 'loss_bbox.txt')) ; 
  rpn_cls_loss = load(fullfile(opts.logDir, 'rpn_cls_loss.txt')) ; 
  rpn_loss_bbox = load(fullfile(opts.logDir, 'rpn_loss_bbox.txt')) ; 
  minibatchesPerIter = 20 ; iter = (1:numel(loss_cls)) * minibatchesPerIter ;

  % load mcn-faster-rcnn training curves
  epochs = 1:opts.numEpochs ;
  checkpointDir = fullfile(vl_rootnn, 'data/pascal', opts.expDir) ;
  snapshot = sprintf('net-epoch-%d.mat', opts.numEpochs) ;
  checkpoint = load(fullfile(checkpointDir, snapshot)) ;
  % compare on the val set to minimse the effect of the long rolling average 
  % across the training set (val set is used during training)
  valStats = checkpoint.stats.val ; 

  mcn_rpn_loss_cls = [valStats.rpn_loss_cls] ;
  mcn_rpn_loss_bbox = [valStats.rpn_loss_bbox] ;
  mcn_loss_cls = [valStats.loss_cls] ;
  mcn_loss_bbox = [valStats.loss_bbox] ;

  switch opts.disp
    case 'caffe'
      smoothK = 50 ; % apply smoothing for easier visualisation
      loss_cls_smooth = movmean(loss_cls, smoothK) ;
      loss_bbox_smooth = movmean(loss_bbox, smoothK) ;
      rpn_cls_loss_smooth = movmean(rpn_cls_loss, smoothK) ;
      rpn_loss_bbox_smooth = movmean(rpn_loss_bbox, smoothK) ;

      clf ; hold all ;
      plot(iter, loss_cls_smooth, 'DisplayName', 'loss-cls') ; 
      plot(iter, loss_bbox_smooth, 'DisplayName', 'loss-bbox') ; 
      plot(iter, rpn_cls_loss_smooth, 'DisplayName', 'rpn-loss-cls') ; 
      plot(iter, rpn_loss_bbox_smooth, 'DisplayName', 'rpn-loss-bbox') ; 
      legend('-DynamicLegend') ; grid ;
      title('Training curves for py-faster-rcnn') ;
      if exist('zs_dispFig', 'file'), zs_dispFig ; end


    case 'mcn'
      clf ; hold all ; % plot matconvnet training curves
      plot(epochs, mcn_loss_cls, 'DisplayName', 'loss-cls') ; 
      plot(epochs, mcn_loss_bbox, 'DisplayName', 'loss-bbox') ; 
      plot(epochs, mcn_rpn_loss_bbox, 'DisplayName', 'rpn-loss-bbox') ; 
      plot(epochs, mcn_rpn_loss_cls, 'DisplayName', 'rpn-loss-cls') ; 
      legend('-DynamicLegend') ; grid ;
      title('Training curves for mcnFasterRCNN') ;
      if exist('zs_dispFig', 'file'), zs_dispFig ; end

    case 'combined'
      % combined plots (apply even heavier smoothing to caffe)
      smoothK = 250 ; % equivalent to approx one epoch
      mcnIter = epochs * 5000 ;
      cls_smooth = movmean(loss_cls, smoothK) ; 
      bbox_smooth = movmean(loss_bbox, smoothK) ;
      rpn_cls_smooth = movmean(rpn_cls_loss, smoothK) ;
      rpn_bbox_smooth = movmean(rpn_loss_bbox, smoothK) ;

      clf ; hold all ;
      plot(iter, cls_smooth, 'r-', 'DisplayName', 'caffe-loss-cls') ; 
      plot(iter, bbox_smooth, 'b-', 'DisplayName', 'caffe-loss-bbox') ; 
      plot(mcnIter, mcn_loss_cls, 'r--', 'DisplayName', 'mcn-loss-cls') ; 
      plot(mcnIter, mcn_loss_bbox, 'b--', 'DisplayName', 'mcn-loss-bbox') ; 

      plot(iter, rpn_cls_smooth, 'g-', 'DisplayName', 'caffe-rpn-loss-cls') ; 
      plot(iter, rpn_bbox_smooth, 'c-', 'DisplayName', 'caffe-rpn-loss-bbox') ; 
      plot(mcnIter, mcn_rpn_loss_cls, 'g--', 'DisplayName', 'mcn-rpn-loss-cls') ; 
      plot(mcnIter, mcn_rpn_loss_bbox, 'c--', 'DisplayName', 'mcn-rpn-loss-bbox') ; 

      legend('-DynamicLegend') ; grid ;
      title(sprintf('caffe vs %s', opts.expDir)) ;
      figPath = fullfile(opts.figDir, sprintf('%s-combined.png', opts.expDir)) ;
      print(figPath, '-dpng') ; if exist('zs_dispFig', 'file'), zs_dispFig ; end
  end
